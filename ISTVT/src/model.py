"""
model.py
========
The Interpretable Spatial-Temporal Video Transformer (ISTVT).

Pipeline (matches the paper, Sec. III):

    sequence (B,T,3,300,300)
        -> Xception entry flow per frame      -> (B,T,728,19,19)
        -> flatten to tokens                  -> (B,T,361,728)
        -> prepend spatial cls token          -> (B,T,362,728)
        -> add learnable position embedding
        -> prepend temporal cls token         -> (B,T+1,362,728)
        -> M decomposed ST-transformer blocks
        -> take token O[:,0,0]                 -> (B,728)
        -> MLP head                            -> (B,1)   (BCE logit)

Each ST block = TemporalResidualAttention (self-subtract) -> SpatialOnlyAttention
               -> FeedForward, all pre-norm with one residual around the two
               attentions (the paper's "decomposed, temporal-first" variant).

The attention modules expose their per-head attention weights and register a
hook on the attention gradient so the interpretability code (visualize.py) can
build separate spatial / temporal relevance heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .xception_backbone import XceptionEntryFlow


# --------------------------------------------------------------------------- #
#  Building blocks                                                             #
# --------------------------------------------------------------------------- #
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _AttnBase(nn.Module):
    """Stores attention map + its gradient for interpretability."""
    def __init__(self):
        super().__init__()
        self.attn = None       # softmax weights, detached graph kept for grad hook
        self.attn_grad = None

    def _save_attn(self, attn):
        self.attn = attn
        if attn.requires_grad:
            attn.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.attn_grad = grad


class SpatialOnlyAttention(_AttnBase):
    """Self-attention within each frame, over its (HW+1) spatial tokens."""
    def __init__(self, dim, heads, dim_head, num_spatial_tokens, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.s = num_spatial_tokens
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):                       # x: (b, (t*s), dim)
        h, s = self.heads, self.s
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b (tt s) (h d) -> b h tt s d', h=h, s=s) for t in (q, k, v))
        dots = torch.einsum('b h t i d, b h t j d -> b h t i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        self._save_attn(attn)
        out = torch.einsum('b h t i j, b h t j d -> b h t i d', attn, v)
        out = rearrange(out, 'b h tt s d -> b (tt s) (h d)')
        return self.to_out(out)


class TemporalResidualAttention(_AttnBase):
    """
    Temporal self-attention with the SELF-SUBTRACT mechanism (paper Eq. 3):
    queries & keys are projected from the temporal residual I' (frame_t - frame_{t-1}),
    values are projected from the original I.  Attention runs over the (T+1)
    temporal tokens at each spatial position.
    """
    def __init__(self, dim, heads, dim_head, num_spatial_tokens, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.s = num_spatial_tokens
        self.to_qk = nn.Linear(dim, inner * 2, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):                       # x: (b, (t*s), dim)
        h, s = self.heads, self.s
        xr = rearrange(x, 'b (tt s) d -> b tt s d', s=s)
        # self-subtract: keep cls-row(0) and frame-0(1); replace frames 2.. with deltas
        residual = torch.cat((xr[:, 0:2], xr[:, 2:] - xr[:, 1:-1]), dim=1)
        residual = rearrange(residual, 'b tt s d -> b (tt s) d')

        q, k = self.to_qk(residual).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k, v = (rearrange(t, 'b (tt s) (h d) -> b h s tt d', h=h, s=s) for t in (q, k, v))
        dots = torch.einsum('b h s i d, b h s j d -> b h s i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        self._save_attn(attn)
        out = torch.einsum('b h s i j, b h s j d -> b h s i d', attn, v)
        out = rearrange(out, 'b h s tt d -> b (tt s) (h d)')
        return self.to_out(out)


class STBlock(nn.Module):
    """temporal(self-subtract) -> spatial -> FFN, pre-norm, decomposed temporal-first."""
    def __init__(self, dim, heads, dim_head, mlp_dim, num_spatial_tokens, dropout=0.0):
        super().__init__()
        self.attn_t = PreNorm(dim, TemporalResidualAttention(dim, heads, dim_head, num_spatial_tokens, dropout))
        self.attn_s = PreNorm(dim, SpatialOnlyAttention(dim, heads, dim_head, num_spatial_tokens, dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(self, x):
        x = self.attn_s(self.attn_t(x)) + x
        x = self.ff(x) + x
        return x


# --------------------------------------------------------------------------- #
#  Full ISTVT                                                                  #
# --------------------------------------------------------------------------- #
class ISTVT(nn.Module):
    def __init__(self, seq_len=6, feature_grid=19, dim=728, depth=12, heads=8,
                 dim_head=64, mlp_scale=4, dropout=0.0, pretrained_backbone=True,
                 num_classes=1):
        super().__init__()
        self.seq_len = seq_len
        self.grid = feature_grid
        self.num_patches = feature_grid * feature_grid            # 361
        self.num_spatial_tokens = self.num_patches + 1            # 362 (+spatial cls)

        self.backbone = XceptionEntryFlow(pretrained=pretrained_backbone)

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, self.num_spatial_tokens, dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            STBlock(dim, heads, dim_head, dim * mlp_scale, self.num_spatial_tokens, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):                        # x: (b, t, 3, 300, 300)
        b, t = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        feat = self.backbone(x)                  # (b*t, 728, 19, 19)
        feat = rearrange(feat, '(b t) c h w -> b t (h w) c', b=b)   # (b,t,361,728)

        cls_space = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space, feat), dim=2)                    # (b,t,362,728)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)

        cls_temporal = repeat(self.temporal_token, '() n d -> b n s d',
                              b=b, s=self.num_spatial_tokens)        # (b,1,362,728)
        x = torch.cat((cls_temporal, x), dim=1)                     # (b,t+1,362,728)

        x = rearrange(x, 'b t s d -> b (t s) d')
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = rearrange(x, 'b (t s) d -> b t s d', s=self.num_spatial_tokens)

        cls_out = x[:, 0, 0]                      # token at (temporal=0, spatial=0)
        return self.mlp_head(cls_out)            # (b, num_classes)

    # convenient access for the interpretability script
    def attention_layers(self):
        spatial, temporal = [], []
        for blk in self.blocks:
            temporal.append(blk.attn_t.fn)
            spatial.append(blk.attn_s.fn)
        return spatial, temporal


def build_model(cfg):
    return ISTVT(
        seq_len=cfg.SEQ_LEN, feature_grid=cfg.FEATURE_GRID, dim=cfg.EMBED_DIM,
        depth=cfg.DEPTH, heads=cfg.NUM_HEADS, dim_head=cfg.DIM_HEAD,
        mlp_scale=cfg.MLP_SCALE, dropout=cfg.DROPOUT,
        pretrained_backbone=cfg.USE_PRETRAINED_BACKBONE, num_classes=1,
    )


if __name__ == "__main__":
    m = ISTVT(seq_len=6, depth=2, pretrained_backbone=False)   # small for a quick test
    out = m(torch.randn(2, 6, 3, 300, 300))
    print("logits:", tuple(out.shape))           # expect (2, 1)
    assert out.shape == (2, 1)
    out.sum().backward()
    print("backward OK")
