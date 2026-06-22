"""
visualize.py
============
ISTVT interpretability (paper Sec. III-D / Algorithm 1): produce SEPARATE
spatial and temporal heatmaps for a face sequence, using gradient-weighted
attention relevance propagation through the decomposed attentions.

For each transformer block we read the stored attention map A and its gradient
dA, form  cam = mean_heads( ReLU(dA * A) ), add the identity, and roll it out
(matrix-multiply) across blocks -- separately for the temporal and spatial
attentions. The classification-token row then gives the per-frame (temporal)
and per-patch (spatial) relevance, upscaled to the input resolution.

    python -m src.visualize --num_videos 4

Light step. Runs on GPU or CPU.
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.model import build_model
from src.dataset import read_manifest, split_rows, _list_frames, _load_window
from src import engine


def _rollout(cams):
    """cams: list of (G, n, n) gradient-weighted attentions. Returns (G, n, n)."""
    n = cams[0].shape[-1]
    g = cams[0].shape[0]
    R = torch.eye(n, device=cams[0].device).unsqueeze(0).expand(g, n, n).clone()
    for cam in cams:
        cam = cam + torch.eye(n, device=cam.device).unsqueeze(0)
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
        R = torch.bmm(cam, R)
    return R


def relevance(model, x):
    """x: (1,T,3,H,W). Returns U_t, U_s each (T,grid,grid) in [0,1]."""
    model.eval()
    model.zero_grad()
    logits = model(x)
    # class 0 = Fake (paper considers Fake-class relevance); logit>0 => fake here,
    # so we backprop the fake logit to highlight forgery evidence.
    logits.sum().backward()

    spatial, temporal = model.attention_layers()
    grid = model.grid
    S = model.num_spatial_tokens          # HW+1
    T1 = model.seq_len + 1                 # T+1

    # ---- temporal: each layer attn (1,h,S,T1,T1) ----
    cams_t = []
    for layer in temporal:
        A, dA = layer.attn, layer.attn_grad
        cam = (dA * A).clamp(min=0).mean(1)          # (1,S,T1,T1)
        cams_t.append(cam.reshape(-1, T1, T1))       # (S,T1,T1)
    Rt = _rollout(cams_t)                            # (S,T1,T1)
    Ut = Rt[:, 0, 1:]                                # cls row, drop temporal cls -> (S,T)
    Ut = Ut[1:, :]                                   # drop spatial cls -> (HW,T)
    Ut = Ut.transpose(0, 1).reshape(model.seq_len, grid, grid)

    # ---- spatial: each layer attn (1,h,T1,S,S) ----
    cams_s = []
    for layer in spatial:
        A, dA = layer.attn, layer.attn_grad
        cam = (dA * A).clamp(min=0).mean(1)          # (1,T1,S,S)
        cams_s.append(cam.reshape(-1, S, S))         # (T1,S,S)
    Rs = _rollout(cams_s)                            # (T1,S,S)
    Us = Rs[:, 0, 1:]                                # cls row, drop spatial cls -> (T1,HW)
    Us = Us[1:, :]                                   # drop temporal cls -> (T,HW)
    Us = Us.reshape(model.seq_len, grid, grid)

    def norm(u):
        u = u.detach().float()
        u = u - u.amin(dim=(1, 2), keepdim=True)
        u = u / (u.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return u
    return norm(Ut), norm(Us)


def overlay(frame_rgb, heat):
    h = cv2.resize(heat, (frame_rgb.shape[1], frame_rgb.shape[0]))
    h = np.uint8(255 * h)
    cmap = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * frame_rgb + 0.45 * cmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num_videos", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(cfg.CKPT_DIR, cfg.RUN_NAME, "best.pt")
    rows = read_manifest(os.path.join(cfg.MANIFEST_DIR, "manifest.csv"))
    test_rows = [r for r in split_rows(rows, "test") if r["label"] == 1][: args.num_videos]
    if not test_rows:
        test_rows = split_rows(rows, "test")[: args.num_videos]

    model = build_model(cfg).to(device)
    engine.load_ckpt(ckpt, model, map_location=device)

    for r in test_rows:
        frames = _list_frames(r["clip_dir"])
        if len(frames) < cfg.SEQ_LEN:
            continue
        x = _load_window(frames, 0, cfg.SEQ_LEN, flip=False).unsqueeze(0).to(device)
        x.requires_grad_(True)
        Ut, Us = relevance(model, x)

        raw = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frames[:cfg.SEQ_LEN]]
        cols = []
        for t in range(cfg.SEQ_LEN):
            top = overlay(raw[t], Us[t].cpu().numpy())     # spatial
            bot = overlay(raw[t], Ut[t].cpu().numpy())     # temporal
            cols.append(np.vstack([raw[t], top, bot]))
        grid_img = np.hstack(cols)
        name = r["video_id"].replace(os.sep, "_")
        out = os.path.join(cfg.VIS_DIR, f"{name}.png")
        cv2.imwrite(out, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print(f"saved {out}  (rows: input / spatial / temporal heatmaps)")


if __name__ == "__main__":
    main()
