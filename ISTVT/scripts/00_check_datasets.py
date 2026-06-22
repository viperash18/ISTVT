"""
00_check_datasets.py
====================
CPU-only sanity check. Verifies your RAW_PATHS, counts videos and class balance
per dataset, and confirms a couple of videos actually open with OpenCV -- BEFORE
you spend any GPU credits.

    python scripts/00_check_datasets.py
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src import preprocess


def main():
    cfg.describe()
    for name in cfg.RAW_PATHS:
        print("\n" + "-" * 60)
        print(f"DATASET: {name}  ({cfg.RAW_PATHS[name]})")
        if not os.path.isdir(cfg.RAW_PATHS[name]):
            print("  PATH MISSING -- fix RAW_PATHS in config.py")
            continue
        try:
            items = preprocess.enumerate_dataset(name)
        except Exception as e:                      # noqa: BLE001
            print(f"  enumeration failed: {e}")
            continue
        n_real = sum(1 for _, _, l in items if l == 0)
        n_fake = sum(1 for _, _, l in items if l == 1)
        print(f"  videos found : {len(items)}  (real={n_real}, fake={n_fake})")
        # probe first openable video
        probed = ok = 0
        for path, _, _ in items[:20]:
            probed += 1
            cap = cv2.VideoCapture(path)
            if cap.isOpened() and cap.read()[0]:
                ok += 1
            cap.release()
        print(f"  openable     : {ok}/{probed} of the first videos decode fine")
        if items:
            print(f"  example      : {items[0][0]}")

    print("\nIf counts look right and at least some videos open, you're ready to "
          "preprocess. If a dataset shows 0 videos, re-check its RAW_PATHS folder.")


if __name__ == "__main__":
    main()
