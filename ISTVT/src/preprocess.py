"""
preprocess.py
=============
Extract aligned 300x300 face crops from the raw videos of any dataset.

Why this design:
  * ROBUST  -- every video is wrapped in try/except; a corrupt / unreadable file
               (FaceForensics++ has many) is logged and skipped, never crashes
               the run.
  * RESUMABLE-- a per-dataset `_processed.json` records finished videos, so if the
               Lightning runtime disconnects you just re-run and it continues.
  * CHEAP   -- only a few short clips per video are sampled (config NUM_CLIPS x
               CLIP_LEN), not every frame, keeping cost and storage modest.

Face crop follows the paper: MTCNN detection, box centered on the NOSE TIP,
side = 1.25 x max(face_width, face_height), resized to 300x300.

Output layout:
    data/faces/<dataset>/<label>/<video_id>/clip<c>/<f>.jpg
    data/faces/<dataset>/_processed.json
    data/faces/<dataset>/_failed.txt

Run (see README for CPU vs GPU advice):
    python -m src.preprocess --dataset celebdf
    python -m src.preprocess --dataset dfdc
    python -m src.preprocess --dataset ff
"""

import os
import sys
import cv2
import glob
import json
import argparse
import traceback

import numpy as np

# allow `python -m src.preprocess` and `python src/preprocess.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# --------------------------------------------------------------------------- #
#  Video enumeration per dataset  -> list of (video_path, video_id, label)     #
#  label: 0 = real, 1 = fake                                                   #
# --------------------------------------------------------------------------- #
def _vid(*parts):
    return "__".join(p.replace(os.sep, "_") for p in parts)


def enumerate_celebdf(root):
    items = []
    spec = [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]
    for sub, label in spec:
        d = os.path.join(root, sub)
        for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            items.append((p, _vid(sub, os.path.splitext(os.path.basename(p))[0]), label))
    return items


def enumerate_dfdc(root):
    """Use every metadata.json found under root (handles train_sample_videos and parts)."""
    items = []
    metas = glob.glob(os.path.join(root, "**", "metadata.json"), recursive=True)
    if not metas:
        print(f"[dfdc] WARNING: no metadata.json under {root}. "
              f"Falling back to folder-name labels.")
        return enumerate_by_keywords(root)
    for meta in metas:
        folder = os.path.dirname(meta)
        with open(meta) as f:
            data = json.load(f)
        for fname, info in data.items():
            p = os.path.join(folder, fname)
            if not os.path.isfile(p):
                continue
            label = 1 if str(info.get("label", "FAKE")).upper() == "FAKE" else 0
            items.append((p, _vid(os.path.basename(folder), os.path.splitext(fname)[0]), label))
    return items


_REAL_KEYS = ("original", "real", "actor", "youtube", "pristine")


def enumerate_by_keywords(root):
    """Generic: label by path. Used for FaceForensics++ and as a DFDC fallback."""
    items = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True)):
        low = p.lower()
        if os.sep + "csv" + os.sep in low:
            continue
        label = 0 if any(k in low for k in _REAL_KEYS) else 1
        rel = os.path.relpath(p, root)
        items.append((p, _vid(os.path.splitext(rel)[0]), label))
    return items


def enumerate_dataset(name):
    root = cfg.RAW_PATHS[name]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Raw path for '{name}' not found: {root}\n"
                                f"Edit RAW_PATHS in config.py.")
    if name == "celebdf":
        return enumerate_celebdf(root)
    if name == "dfdc":
        return enumerate_dfdc(root)
    return enumerate_by_keywords(root)            # ff and anything else


# --------------------------------------------------------------------------- #
#  Face cropping                                                               #
# --------------------------------------------------------------------------- #
def make_mtcnn():
    import torch
    from facenet_pytorch import MTCNN
    dev = cfg.MTCNN_DEVICE if (cfg.MTCNN_DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != cfg.MTCNN_DEVICE:
        print(f"[mtcnn] CUDA not available -> running MTCNN on CPU.")
    # keep_all=False -> return the single most probable face
    return MTCNN(keep_all=True, device=dev, post_process=False, select_largest=True), dev


def crop_face(frame_rgb, box, landmarks):
    """Nose-centered, 1.25x max-side square crop -> FACE_SIZE x FACE_SIZE BGR uint8."""
    H, W = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    fw, fh = (x2 - x1), (y2 - y1)
    side = cfg.BOX_SCALE * max(fw, fh)
    if landmarks is not None:
        cx, cy = landmarks[2]                     # nose tip
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    L, T = int(round(cx - half)), int(round(cy - half))
    R, B = int(round(cx + half)), int(round(cy + half))
    # pad if the box runs off the image so the face stays centered
    padL, padT = max(0, -L), max(0, -T)
    padR, padB = max(0, R - W), max(0, B - H)
    if padL or padT or padR or padB:
        frame_rgb = cv2.copyMakeBorder(frame_rgb, padT, padB, padL, padR,
                                       cv2.BORDER_REFLECT)
        L += padL; R += padL; T += padT; B += padT
    crop = frame_rgb[T:B, L:R]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (cfg.FACE_SIZE, cfg.FACE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def sample_clip_starts(total_frames, num_clips, clip_len):
    if total_frames <= clip_len:
        return [0]
    last = total_frames - clip_len
    if num_clips == 1:
        return [last // 2]
    step = last / (num_clips - 1)
    return sorted(set(int(round(i * step)) for i in range(num_clips)))


def read_clip(cap, start, clip_len):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(clip_len):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def process_video(path, video_id, label, out_root, mtcnn):
    """Returns number of clips successfully written (>=1 means usable)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:                                # some corrupt files report 0
        total = cfg.NUM_CLIPS * cfg.CLIP_LEN * 4

    starts = sample_clip_starts(total, cfg.NUM_CLIPS, cfg.CLIP_LEN)
    out_dir = os.path.join(out_root, str(label), video_id)
    written_clips = 0

    for ci, start in enumerate(starts):
        frames = read_clip(cap, start, cfg.CLIP_LEN)
        if len(frames) < cfg.SEQ_LEN:
            continue
        boxes, probs, lms = mtcnn.detect(frames, landmarks=True)
        clip_dir = os.path.join(out_dir, f"clip{ci}")
        saved = 0
        for fi, (frame, box, prob, lm) in enumerate(zip(frames, boxes, probs, lms)):
            if box is None or prob is None or prob[0] is None or prob[0] < cfg.MIN_FACE_PROB:
                continue
            face = crop_face(frame, box[0], None if lm is None else lm[0])
            if face is None:
                continue
            os.makedirs(clip_dir, exist_ok=True)
            cv2.imwrite(os.path.join(clip_dir, f"{saved:02d}.jpg"), face,
                        [int(cv2.IMWRITE_JPEG_QUALITY), cfg.JPEG_QUALITY])
            saved += 1
        if saved >= cfg.SEQ_LEN:
            written_clips += 1
        elif os.path.isdir(clip_dir):             # not enough faces -> drop partial clip
            for f in glob.glob(os.path.join(clip_dir, "*.jpg")):
                os.remove(f)
            os.rmdir(clip_dir)
    cap.release()
    return written_clips


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
def run(dataset):
    out_root = os.path.join(cfg.FACES_DIR, dataset)
    os.makedirs(out_root, exist_ok=True)
    done_path = os.path.join(out_root, "_processed.json")
    fail_path = os.path.join(out_root, "_failed.txt")

    done = {}
    if os.path.isfile(done_path):
        with open(done_path) as f:
            done = json.load(f)

    items = enumerate_dataset(dataset)
    if cfg.MAX_VIDEOS_PER_DATASET > 0:
        import random as _random
        rng = _random.Random(cfg.SEED)
        reals = [it for it in items if it[2] == 0]
        fakes = [it for it in items if it[2] == 1]
        rng.shuffle(reals)
        rng.shuffle(fakes)
        half = cfg.MAX_VIDEOS_PER_DATASET // 2
        keep = reals[:half] + fakes[:half]
        if len(keep) < cfg.MAX_VIDEOS_PER_DATASET:
            extra = reals[half:] + fakes[half:]
            rng.shuffle(extra)
            keep += extra[: cfg.MAX_VIDEOS_PER_DATASET - len(keep)]
        rng.shuffle(keep)
        items = keep
        n_r = sum(1 for it in items if it[2] == 0)
        n_f = sum(1 for it in items if it[2] == 1)
        print(f"[{dataset}] capped to {len(items)} videos (real={n_r}, fake={n_f}).")
    print(f"[{dataset}] {len(items)} videos found, {len(done)} already processed.")

    mtcnn, dev = make_mtcnn()
    print(f"[{dataset}] MTCNN device = {dev}")

    n_ok = n_skip = n_fail = 0
    for i, (path, vid, label) in enumerate(items):
        if vid in done:
            n_skip += 1
            continue
        try:
            clips = process_video(path, vid, label, out_root, mtcnn)
            done[vid] = {"label": label, "clips": clips}
            n_ok += 1 if clips > 0 else 0
            if clips == 0:
                with open(fail_path, "a") as f:
                    f.write(f"{path}\tNO_USABLE_CLIP\n")
        except Exception as e:                    # noqa: BLE001
            n_fail += 1
            with open(fail_path, "a") as f:
                f.write(f"{path}\t{type(e).__name__}: {e}\n")
            traceback.print_exc(limit=1)
        # checkpoint progress every 25 videos (cheap insurance against disconnects)
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            with open(done_path, "w") as f:
                json.dump(done, f)
            print(f"[{dataset}] {i+1}/{len(items)}  ok={n_ok} skip={n_skip} fail={n_fail}")

    with open(done_path, "w") as f:
        json.dump(done, f)
    print(f"[{dataset}] DONE. usable={n_ok} skipped={n_skip} failed={n_fail}")
    print(f"[{dataset}] failures (if any) logged in {fail_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(cfg.RAW_PATHS.keys()))
    args = ap.parse_args()
    cfg.describe()
    run(args.dataset)
