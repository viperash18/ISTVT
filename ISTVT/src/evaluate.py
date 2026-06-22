"""
evaluate.py
===========
Evaluate the trained model on the UNSEEN cross-dataset (config.CROSS_TEST_DATASET).
Reports video-level AUROC and accuracy -- the paper's cross-dataset protocol.

    python -m src.evaluate                 # uses best.pt
    python -m src.evaluate --ckpt path.pt  # any checkpoint

Light GPU step (inference only). A T4 is plenty; CPU works but is slow.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.model import build_model
from src.dataset import read_manifest, split_rows, SequenceDataset
from src import engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: run best.pt)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    cfg.describe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt or os.path.join(cfg.CKPT_DIR, cfg.RUN_NAME, "best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first.")

    rows = read_manifest(os.path.join(cfg.MANIFEST_DIR, "manifest.csv"))
    eval_rows = split_rows(rows, args.split)
    if not eval_rows:
        raise RuntimeError(f"No '{args.split}' clips in manifest. "
                           f"Check preprocessing for '{cfg.CROSS_TEST_DATASET}'.")
    ds = SequenceDataset(eval_rows, cfg.SEQ_LEN, train=False,
                         seqs_per_video_eval=cfg.SEQUENCES_PER_VIDEO_EVAL)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = build_model(cfg).to(device)
    engine.load_ckpt(ckpt, model, map_location=device)
    print(f"loaded {ckpt}")

    metrics, scores, labels = engine.evaluate(
        model, loader, device, use_amp=cfg.USE_AMP and device.type == "cuda")

    target = cfg.CROSS_TEST_DATASET if args.split == "test" else "in-domain val"
    print("\n" + "=" * 60)
    print(f"CROSS-DATASET EVALUATION on '{target}'")
    print("=" * 60)
    print(f"  videos     : {metrics['n_videos']}")
    print(f"  AUROC      : {metrics['auc']*100:.2f}%")
    print(f"  accuracy   : {metrics['acc']*100:.2f}%  (threshold 0.5)")
    print(f"  best-thr acc:{metrics['best_acc']*100:.2f}%")
    print("=" * 60)

    out = os.path.join(cfg.LOG_DIR, cfg.RUN_NAME + f"_{args.split}_results.json")
    json.dump({"metrics": metrics,
               "video_scores": {v: float(np.mean(s)) for v, s in scores.items()},
               "video_labels": labels}, open(out, "w"), indent=2)
    print(f"saved per-video scores -> {out}")


if __name__ == "__main__":
    main()
