"""
engine.py
=========
Reusable training / evaluation logic: one-epoch loop, video-level metric
computation, and checkpoint save/load (so a disconnect never costs an epoch).
"""

import os
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.metrics import roc_auc_score
except Exception:                                   # noqa: BLE001
    roc_auc_score = None


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _roc_auc(labels, probs):
    """AUROC via Mann-Whitney U (pure NumPy, handles ties). No sklearn needed."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty(len(probs), dtype=float)
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def video_level_metrics(video_scores, video_labels):
    """video_scores/labels: dict video_id -> list[prob] / label. Returns dict."""
    vids = list(video_scores.keys())
    probs = np.array([np.mean(video_scores[v]) for v in vids])
    labels = np.array([video_labels[v] for v in vids])
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    auc = _roc_auc(labels, probs) if len(labels) else float("nan")
    # best-threshold accuracy (useful under class imbalance / domain shift)
    best_acc = acc
    if len(labels):
        for t in np.linspace(0.05, 0.95, 19):
            a = float(((probs >= t).astype(int) == labels).mean())
            best_acc = max(best_acc, a)
    return {"auc": auc, "acc": acc, "best_acc": best_acc, "n_videos": len(labels)}


# --------------------------------------------------------------------------- #
#  Schedule                                                                    #
# --------------------------------------------------------------------------- #
def lr_at(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


# --------------------------------------------------------------------------- #
#  Train / eval loops                                                          #
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, scaler, device, criterion,
                    epoch, total_steps, warmup_steps, base_lr, global_step,
                    grad_accum=1, grad_clip=1.0, use_amp=True, log_every=50):
    model.train()
    running = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y, _) in enumerate(loader):
        lr = lr_at(global_step, total_steps, warmup_steps, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=use_amp):
            logits = model(x).view(-1)
            loss = criterion(logits, y) / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0:
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        running += loss.item() * grad_accum * x.size(0)
        seen += x.size(0)
        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch} step {i+1}/{len(loader)} "
                  f"lr={lr:.2e} loss={running/max(1,seen):.4f}")
    return running / max(1, seen), global_step


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    scores = defaultdict(list)
    labels = {}
    for x, y, vids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=use_amp):
            prob = torch.sigmoid(model(x).view(-1)).float().cpu().numpy()
        for p, lab, v in zip(prob, y.numpy(), vids):
            scores[v].append(float(p))
            labels[v] = int(lab)
    return video_level_metrics(scores, labels), scores, labels


# --------------------------------------------------------------------------- #
#  Checkpoints                                                                 #
# --------------------------------------------------------------------------- #
def save_ckpt(path, model, optimizer, scaler, epoch, global_step, best_auc, extra=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_auc": best_auc,
        "extra": extra or {},
    }, path)


def load_ckpt(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    return ck
