"""
train.py
========
Train ISTVT on the pooled TRAIN_DATASETS and validate on the held-out in-domain
val split. Auto-resumes from the last checkpoint, so if Lightning disconnects you
just run the same command again.

    python -m src.train

GPU required (this is the heavy step). Use an A10G/A100 on Lightning AI.
"""

import os
import sys
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.model import build_model
from src.dataset import read_manifest, split_rows, SequenceDataset
from src import engine


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.LEARNING_RATE,
                               momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
    return torch.optim.AdamW(params, lr=cfg.LEARNING_RATE,
                             weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))


def main():
    cfg.describe()
    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU detected. Training on CPU will be extremely slow. "
              "Switch the Lightning Studio to a GPU machine.")
    torch.backends.cudnn.benchmark = True

    manifest = os.path.join(cfg.MANIFEST_DIR, "manifest.csv")
    rows = read_manifest(manifest)
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    if not train_rows:
        raise RuntimeError("No training clips in manifest. Run preprocessing + build_manifest.")
    print(f"train clips={len(train_rows)}  val clips={len(val_rows)}")

    train_ds = SequenceDataset(train_rows, cfg.SEQ_LEN, train=True,
                               seqs_per_video_train=cfg.SEQUENCES_PER_VIDEO_TRAIN,
                               aug=True)
    val_ds = SequenceDataset(val_rows, cfg.SEQ_LEN, train=False,
                             seqs_per_video_eval=cfg.SEQUENCES_PER_VIDEO_EVAL)

    sampler = train_ds.class_balanced_sampler() if cfg.BALANCE_CLASSES else None
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              sampler=sampler, shuffle=sampler is None,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=cfg.NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True,
                            persistent_workers=cfg.NUM_WORKERS > 0)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()

    run_dir = os.path.join(cfg.CKPT_DIR, cfg.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    hist_path = os.path.join(cfg.LOG_DIR, cfg.RUN_NAME + "_history.json")

    steps_per_epoch = max(1, len(train_loader) // cfg.GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * cfg.EPOCHS
    warmup_steps = steps_per_epoch * cfg.WARMUP_EPOCHS

    start_epoch, global_step, best_auc = 0, 0, -1.0
    history = []
    if cfg.RESUME and os.path.isfile(last_path):
        ck = engine.load_ckpt(last_path, model, optimizer, scaler, map_location=device)
        start_epoch = ck["epoch"] + 1
        global_step = ck.get("global_step", 0)
        best_auc = ck.get("best_auc", -1.0)
        if os.path.isfile(hist_path):
            history = json.load(open(hist_path))
        print(f"RESUMED from {last_path}: next epoch {start_epoch}, best_auc={best_auc:.4f}")

    for epoch in range(start_epoch, cfg.EPOCHS):
        tr_loss, global_step = engine.train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion,
            epoch, total_steps, warmup_steps, cfg.LEARNING_RATE, global_step,
            grad_accum=cfg.GRAD_ACCUM_STEPS, grad_clip=cfg.GRAD_CLIP,
            use_amp=cfg.USE_AMP and device.type == "cuda")

        metrics, _, _ = engine.evaluate(model, val_loader, device,
                                        use_amp=cfg.USE_AMP and device.type == "cuda")
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f}  "
              f"val_auc={metrics['auc']:.4f} val_acc={metrics['acc']:.4f} "
              f"val_best_acc={metrics['best_acc']:.4f} (n={metrics['n_videos']})")

        history.append({"epoch": epoch, "train_loss": tr_loss, **metrics})
        json.dump(history, open(hist_path, "w"), indent=2)

        engine.save_ckpt(last_path, model, optimizer, scaler, epoch,
                         global_step, best_auc, extra={"val": metrics})
        score = metrics["auc"] if not np.isnan(metrics["auc"]) else metrics["best_acc"]
        if score > best_auc:
            best_auc = score
            engine.save_ckpt(best_path, model, optimizer, scaler, epoch,
                             global_step, best_auc, extra={"val": metrics})
            print(f"  -> new best ({best_auc:.4f}) saved to {best_path}")

    print(f"\nTraining done. Best val score: {best_auc:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
