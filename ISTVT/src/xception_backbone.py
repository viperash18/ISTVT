"""
xception_backbone.py
====================
Feature extractor for ISTVT: the *entry flow* of the Xception network.

A 300x300x3 face goes in, a 19x19x728 feature map comes out -- exactly the
"tiny convolutional network composed of several Xception blocks (the entry flow
of Xception)" described in the paper.

We load ImageNet-pretrained weights from `timm` when available (best for
accuracy) and fall back to a clean from-scratch entry flow if `timm` or the
download is unavailable, so the code always runs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Vendored, dependency-free Xception entry flow (used as the fallback)        #
# --------------------------------------------------------------------------- #
class SeparableConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, k, stride, padding, dilation,
                               groups=in_c, bias=bias)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_f != in_f or strides != 1:
            self.skip = nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)
        else:
            self.skip = None

        rep = []
        filters = in_f
        if grow_first:
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(in_f, out_f, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_f)]
            filters = out_f
        for _ in range(reps - 1):
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(in_f, out_f, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_f)]
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        skip = inp if self.skip is None else self.skipbn(self.skip(inp))
        return x + skip


class _VendoredEntryFlow(nn.Module):
    """conv1 -> conv2 -> block1 -> block2 -> block3  ==> 19x19x728 for 300x300 in."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# --------------------------------------------------------------------------- #
#  Public backbone                                                             #
# --------------------------------------------------------------------------- #
class XceptionEntryFlow(nn.Module):
    """
    Returns a 19x19x728 feature map from a 300x300x3 image.

    Tries (in order):
      1. timm 'xception'         (pretrained, best accuracy)
      2. timm 'legacy_xception'  (pretrained)
      3. vendored entry flow     (random init -- always works)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.out_channels = 728
        self.backbone = None
        self.source = "vendored(random-init)"

        if pretrained:
            self.backbone, self.source = self._try_timm()

        if self.backbone is None:
            self.backbone = _VendoredEntryFlow()

        print(f"[backbone] Xception entry flow loaded from: {self.source}")

    @staticmethod
    def _try_timm():
        try:
            import timm
        except Exception as e:                      # noqa: BLE001
            print(f"[backbone] timm not available ({e}); using random-init entry flow.")
            return None, "vendored(random-init)"

        for name in ("xception", "legacy_xception"):
            try:
                m = timm.create_model(name, pretrained=True)
            except Exception as e:                  # noqa: BLE001
                print(f"[backbone] timm.create_model('{name}', pretrained=True) failed: {e}")
                continue
            entry = XceptionEntryFlow._slice_entry(m)
            if entry is not None:
                return entry, f"timm:{name}(pretrained)"
        return None, "vendored(random-init)"

    @staticmethod
    def _slice_entry(m):
        """Pull conv1..block3 out of a timm Xception, tolerant to attr naming."""
        wanted = ["conv1", "bn1", "act1", "conv2", "bn2", "act2",
                  "block1", "block2", "block3"]
        mods = []
        for n in wanted:
            if hasattr(m, n):
                mods.append(getattr(m, n))
            elif n.startswith("act"):
                mods.append(nn.ReLU(inplace=True))   # some versions drop explicit act
            else:
                return None                          # missing a real layer -> bail out
        return nn.Sequential(*mods)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    net = XceptionEntryFlow(pretrained=False)
    y = net(torch.randn(2, 3, 300, 300))
    print("entry-flow output:", tuple(y.shape))     # expect (2, 728, 19, 19)
    assert y.shape[1:] == (728, 19, 19), y.shape
    print("OK")
