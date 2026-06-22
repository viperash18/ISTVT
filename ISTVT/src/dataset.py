"""
dataset.py
==========
Reads the manifest and serves SEQ_LEN-frame face sequences to the model.

  * train : each item is one clip; a random window of SEQ_LEN consecutive
            frames is drawn, with a single (consistent) horizontal flip applied
            to the whole sequence. Clips are repeated SEQUENCES_PER_VIDEO_TRAIN
            times per epoch.
  * eval  : each clip is expanded into up to SEQUENCES_PER_VIDEO_EVAL fixed
            windows; every item carries its video_id so evaluate.py can average
            sequence scores into a VIDEO-LEVEL prediction (the paper's protocol).

Pixels are normalised to [-1, 1] (mean=std=0.5), matching the Xception backbone.
"""

import os
import csv
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def read_manifest(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            r["num_frames"] = int(r["num_frames"])
            rows.append(r)
    return rows


def _list_frames(clip_dir):
    return sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))


def _load_window(frame_paths, start, seq_len, flip):
    seq = []
    for p in frame_paths[start:start + seq_len]:
        img = cv2.imread(p)                         # BGR uint8
        if img is None:
            img = np.zeros((300, 300, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip:
            img = img[:, ::-1, :]
        seq.append(img)
    arr = np.stack(seq).astype(np.float32) / 255.0   # (T,H,W,3) in [0,1]
    arr = (arr - 0.5) / 0.5                           # -> [-1,1]
    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))  # (T,3,H,W)
    return torch.from_numpy(arr)


class SequenceDataset(Dataset):
    def __init__(self, rows, seq_len, train, seqs_per_video_train=4,
                 seqs_per_video_eval=6, aug=True):
        self.seq_len = seq_len
        self.train = train
        self.aug = aug and train
        self.items = []                              # (clip_dir, label, video_id, start|-1)

        if train:
            for r in rows:
                for _ in range(seqs_per_video_train):
                    self.items.append((r["clip_dir"], r["label"], r["video_id"], -1))
        else:
            for r in rows:
                frames = _list_frames(r["clip_dir"])
                n = len(frames)
                if n < seq_len:
                    continue
                last = n - seq_len
                k = min(seqs_per_video_eval, last + 1)
                starts = [0] if k <= 1 else [int(round(i * last / (k - 1))) for i in range(k)]
                for s in sorted(set(starts)):
                    self.items.append((r["clip_dir"], r["label"], r["video_id"], s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        clip_dir, label, vid, start = self.items[idx]
        frames = _list_frames(clip_dir)
        n = len(frames)
        if n < self.seq_len:                          # safety pad
            frames = (frames + frames * self.seq_len)[: self.seq_len]
            n = len(frames)
        if start < 0:                                 # train: random window
            start = random.randint(0, max(0, n - self.seq_len))
        flip = self.aug and (random.random() < 0.5)
        x = _load_window(frames, start, self.seq_len, flip)
        return x, torch.tensor(label, dtype=torch.float32), vid

    def class_balanced_sampler(self):
        labels = np.array([it[1] for it in self.items])
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        weights = w[labels]
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                     num_samples=len(self.items), replacement=True)


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]
