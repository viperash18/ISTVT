"""
build_manifest.py
=================
Scan the extracted face crops and produce a single manifest CSV that the
training / evaluation code reads.

One row = one usable clip (a folder of >= SEQ_LEN consecutive face crops).

Columns: dataset, video_id, label, split, clip_dir, num_frames

Splits:
  * Training datasets (config.TRAIN_DATASETS) are split by VIDEO into
    train / val (no video appears in both -> no leakage). Real & fake are
    split separately so val keeps both classes.
  * The cross-test dataset (config.CROSS_TEST_DATASET) -> every clip is 'test'.

Idempotent: just re-run it any time after (re-)preprocessing.

    python -m src.build_manifest
"""

import os
import sys
import csv
import glob
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def scan_clips(dataset):
    """Yield (video_id, label, clip_dir, num_frames) for every usable clip."""
    root = os.path.join(cfg.FACES_DIR, dataset)
    rows = []
    for label in ("0", "1"):
        ldir = os.path.join(root, label)
        if not os.path.isdir(ldir):
            continue
        for video_id in sorted(os.listdir(ldir)):
            vdir = os.path.join(ldir, video_id)
            if not os.path.isdir(vdir):
                continue
            for clip_dir in sorted(glob.glob(os.path.join(vdir, "clip*"))):
                n = len(glob.glob(os.path.join(clip_dir, "*.jpg")))
                if n >= cfg.SEQ_LEN:
                    rows.append((video_id, int(label), clip_dir, n))
    return rows


def split_videos(rows, val_split, seed):
    """Group rows by video_id, hold out val_split of videos per class for validation."""
    by_vid = {}
    for r in rows:
        by_vid.setdefault((r[1], r[0]), []).append(r)   # key = (label, video_id)
    reals = [v for (lab, _), v in by_vid.items() if lab == 0]
    fakes = [v for (lab, _), v in by_vid.items() if lab == 1]
    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)

    def cut(groups):
        k = max(1, int(round(len(groups) * val_split))) if groups else 0
        return groups[k:], groups[:k]             # train_videos, val_videos

    tr_r, va_r = cut(reals)
    tr_f, va_f = cut(fakes)
    train = [r for g in (tr_r + tr_f) for r in g]
    val = [r for g in (va_r + va_f) for r in g]
    return train, val


def main():
    cfg.describe()
    out_csv = os.path.join(cfg.MANIFEST_DIR, "manifest.csv")
    all_rows = []
    counts = {}

    # training datasets -> train/val
    for ds in cfg.TRAIN_DATASETS:
        rows = scan_clips(ds)
        train, val = split_videos(rows, cfg.VAL_SPLIT, cfg.SEED)
        for (vid, lab, cd, n) in train:
            all_rows.append([ds, vid, lab, "train", cd, n])
        for (vid, lab, cd, n) in val:
            all_rows.append([ds, vid, lab, "val", cd, n])
        counts[ds] = {"train_clips": len(train), "val_clips": len(val)}

    # cross-test dataset -> test
    ds = cfg.CROSS_TEST_DATASET
    rows = scan_clips(ds)
    for (vid, lab, cd, n) in rows:
        all_rows.append([ds, vid, lab, "test", cd, n])
    counts[ds] = {"test_clips": len(rows)}

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "video_id", "label", "split", "clip_dir", "num_frames"])
        w.writerows(all_rows)

    # human-readable summary
    print("\n=== manifest summary ===")
    for split in ("train", "val", "test"):
        sel = [r for r in all_rows if r[3] == split]
        n_real = sum(1 for r in sel if r[2] == 0)
        n_fake = sum(1 for r in sel if r[2] == 1)
        print(f"{split:5s}: {len(sel):6d} clips  (real={n_real}, fake={n_fake})")
    print("per-dataset:", counts)
    print(f"\nwrote {out_csv}  ({len(all_rows)} rows)")
    if not any(r[3] == "train" for r in all_rows):
        print("WARNING: no training clips found -- did preprocessing run for "
              f"{cfg.TRAIN_DATASETS}?")
    if not any(r[3] == "test" for r in all_rows):
        print("WARNING: no test clips found -- preprocessing for "
              f"'{cfg.CROSS_TEST_DATASET}' may have produced nothing (corrupt videos?). "
              "Consider swapping CROSS_TEST_DATASET in config.py.")


if __name__ == "__main__":
    main()
