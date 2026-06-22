# ISTVT — Interpretable Spatial-Temporal Video Transformer for Deepfake Detection
### A complete, re-runnable reimplementation built for **Lightning AI Studio**

This project reimplements the paper *"ISTVT: Interpretable Spatial-Temporal Video
Transformer for Deepfake Detection"* (IEEE TIFS, 2023) from scratch in PyTorch and
packages it so a **beginner** can run the whole thing end-to-end on Lightning AI.

It is designed around three real-world facts about your setup:

1. **You will re-run it 2–3 times (or more).** Every heavy step is *checkpointed and
   resumable* — if the runtime disconnects, you just run the same command again and it
   continues from where it stopped. Nothing is ever redone from scratch.
2. **Your FaceForensics++ download is mostly corrupt.** Preprocessing *skips* any video
   it cannot open and logs it, so corrupt files never crash a run. The default plan
   trains on your two **clean** datasets and uses the partly-corrupt one only for the
   unseen cross-test (where missing videos hurt the least).
3. **GPU credits cost money.** Every step below is labelled **CPU** or **GPU** so you
   only switch on the GPU when it actually helps.

---

## 0. The 10-second mental model

Deepfake detection here is a pipeline of small, separate steps. You run them **one at a
time**, in order. Each one writes its results to disk, so the next step just reads what
the previous one produced:

```
raw videos
   │  (preprocess.py)      detect + crop faces  ->  data/faces/...
   ▼
face crops
   │  (build_manifest.py)  list every clip       ->  data/manifests/manifest.csv
   ▼
manifest
   │  (train.py)           train the transformer ->  checkpoints/.../best.pt
   ▼
trained model
   │  (evaluate.py)        cross-dataset AUROC    ->  logs/...scores.json
   │  (visualize.py)       attention heatmaps     ->  visualizations/*.png
   ▼
results
```

You never edit code to run it. You only ever edit **`config.py`** (mostly just the three
dataset paths) and then run commands.

---

## 1. The "train on 2, cross-test on 1" idea (your accuracy goal)

The hardest, most meaningful test for a deepfake detector is **cross-dataset**: train on
some datasets, then test on a *completely different* one the model has never seen. This
is what the paper reports and what you asked for.

Default plan (already set in `config.py`):

| Role | Datasets |
|------|----------|
| **Train + in-domain validation** | Celeb-DF **+** DFDC (your two clean datasets) |
| **Unseen cross-dataset test** | FaceForensics++ (the corrupt one — used only for testing) |

To swap the combination later, change just two lines near the top of `config.py`:

```python
TRAIN_DATASETS = ["celebdf", "dfdc"]   # pooled for training
CROSS_TEST_DATASET = "ff"              # held out, never seen in training
```

> **Honest expectation.** Cross-dataset deepfake accuracy is *genuinely hard* — even the
> paper's numbers drop a lot versus same-dataset testing. Good cross-dataset AUROC is
> roughly in the 0.75–0.95 range depending on the pair; do not expect 99%. The biggest
> single lever you control is **how much clean data you preprocess** and **how many
> epochs you train**. More on tuning at the very bottom.

---

## 2. One-time Lightning AI setup (all **CPU** — costs almost nothing)

### 2.1 Create the Studio
1. Log in to **lightning.ai** → **Studios** → **New Studio**.
2. Pick the **CPU** machine to start (you switch to GPU only for the heavy steps).
   Your Studio home directory is **persistent** — files survive disconnects and
   machine switches, which is exactly why we keep everything under it.

### 2.2 Upload this project
- Drag the whole `istvt_lightning/` folder into the Studio file browser (left panel),
  or upload the zip and unzip it in the terminal:
  ```bash
  unzip istvt_lightning.zip -d ~/   # then: cd ~/istvt_lightning
  ```

### 2.3 Upload the datasets
Put each dataset somewhere under your Studio home. The defaults in `config.py` expect:

```
/teamspace/studios/this_studio/datasets/celeb-df              <- Celeb-DF (v2)
/teamspace/studios/this_studio/datasets/dfdc                  <- DFDC sample
/teamspace/studios/this_studio/datasets/FaceForensics++_C23   <- FaceForensics++
```

- **Celeb-DF** folder must directly contain: `Celeb-real/`, `Celeb-synthesis/`,
  `YouTube-real/`, `List_of_testing_videos.txt`.
- **DFDC** folder must directly contain `train_sample_videos/` (with `metadata.json`
  inside it).
- **FaceForensics++** folder must contain the manipulation subfolders
  (`DeepFakeDetection/`, `original/`, `Deepfakes/`, …). Corrupt ones are fine — they'll
  be skipped.

> Big uploads: the easiest beginner route is Lightning's **drag-and-drop** upload. If a
> dataset is on Google Drive, you can also use the Studio terminal with `gdown`. Uploads
> run on the **CPU** machine — no GPU needed.

If your paths differ, either edit `RAW_PATHS` in `config.py`, **or** set environment
variables before running (no file editing):
```bash
export CELEBDF_DIR=/teamspace/studios/this_studio/datasets/celeb-df
export DFDC_DIR=/teamspace/studios/this_studio/datasets/dfdc
export FF_DIR=/teamspace/studios/this_studio/datasets/FaceForensics++_C23
```

### 2.4 Install the Python packages (**CPU**)
```bash
cd ~/istvt_lightning
pip install -r requirements.txt
```
This installs PyTorch, einops, timm, facenet-pytorch (MTCNN), OpenCV, scikit-learn, etc.

---

## 3. The run order (with **CPU / GPU** for every step)

Here is the entire project as a command list. The **Machine** column is the whole point —
switch the Studio to GPU **only** for the rows that say GPU, then switch back to CPU to
stop burning credits.

| # | Command | Machine | Why |
|---|---------|---------|-----|
| 1 | `python config.py` | **CPU** | Prints the active config + checks dataset paths exist. |
| 2 | `python scripts/00_check_datasets.py` | **CPU** | Counts videos, checks class balance, opens a few to confirm they're readable. |
| 3 | `python -m src.preprocess --dataset celebdf` | **GPU** (T4) recommended | Face detection (MTCNN) is ~10–30× faster on GPU. One-time. |
| 4 | `python -m src.preprocess --dataset dfdc` | **GPU** (T4) | Same as above. |
| 5 | `python -m src.preprocess --dataset ff` | **GPU** (T4) | Same; corrupt videos are auto-skipped + logged. |
| 6 | `python -m src.build_manifest` | **CPU** | Pure file listing → `manifest.csv`. No compute. |
| 7 | `python -m src.train` | **GPU** (A10G / A100) | The actual model training. The expensive part. |
| 8 | `python -m src.evaluate` | **GPU** (T4) or **CPU** | Cross-dataset AUROC. Inference only — light. |
| 9 | `python -m src.visualize` | **GPU** or **CPU** | Attention heatmaps for a few videos. Light. |

> **How to switch machine on Lightning AI:** top-right of the Studio there's a machine
> selector. Click it → choose CPU or a GPU (T4 is the cheap one; A10G/A100 for training).
> Switching keeps all your files. **Switch to GPU right before step 3, and switch back to
> CPU after step 5; switch to GPU again for step 7.**

---

## 4. What each step does, and what "done" looks like

### Step 1 — `python config.py`  · **CPU**
Prints every setting and shows `OK`/`MISSING` next to each dataset path. **If any path
says MISSING, fix it before continuing** (edit `RAW_PATHS` or set the env vars in 2.3).

### Step 2 — `python scripts/00_check_datasets.py`  · **CPU**
Sanity check. For each dataset it reports how many real vs fake videos it found and tries
to open the first ~20 with OpenCV. You want non-zero counts for `celebdf` and `dfdc`. For
`ff` it's normal to see many that won't open — that's the corruption, and it's fine.

### Steps 3–5 — `python -m src.preprocess --dataset <name>`  · **GPU recommended**
Finds faces in each video with MTCNN, crops them **centered on the nose at 1.25× the face
box** and resizes to **300×300** (exactly as the paper specifies), then saves them as JPGs
under `data/faces/<dataset>/<label>/<video_id>/clipX/`.

- **Resumable:** it keeps a `_processed.json` per dataset. If the runtime drops, just run
  the *same* command again — already-done videos are skipped instantly.
- **Crash-proof:** any video that fails to open (corrupt FF++ files!) is caught, written
  to `_failed.txt`, and skipped. The run keeps going.
- Run the three datasets **one at a time**. Each prints progress and a final count.

> This is the one-time cost. After it's done you never touch the raw videos again, and
> you can switch back to CPU.

### Step 6 — `python -m src.build_manifest`  · **CPU**
Walks the saved face crops and writes `data/manifests/manifest.csv` — one row per usable
clip. It splits the **training** datasets by *video* (so no frames from one video leak
across train/val) into `train` / `val`, and marks every clip of the **cross-test**
dataset as `test`. Prints a summary table. If a split is empty it warns you loudly.

### Step 7 — `python -m src.train`  · **GPU (A10G/A100)**
Trains the ISTVT model. Key things it does for you:
- **Auto-resume:** writes `checkpoints/<run>/last.pt` every epoch. Re-run the same
  command after any disconnect and it picks up at the next epoch. `best.pt` tracks the
  best validation AUROC.
- **Class balancing:** Celeb-DF is heavily fake-skewed; a weighted sampler keeps each
  batch balanced so the model can't cheat by always saying "fake".
- **Mixed precision (AMP):** ~2× faster and ~half the memory on GPU.
- Prints train loss + validation AUROC/accuracy each epoch; history saved in `logs/`.

### Step 8 — `python -m src.evaluate`  · **GPU (light) or CPU**
Loads `best.pt` and runs the **unseen** cross-dataset test, aggregating clip scores to a
**video-level** score (as the paper does) and reporting **AUROC** + accuracy. Per-video
scores are saved to `logs/`. Options: `--split val` to instead score the in-domain
validation set, or `--ckpt <path>` to evaluate a specific checkpoint.

### Step 9 — `python -m src.visualize`  · **GPU or CPU**
Produces the paper's **interpretability** output: gradient-weighted attention rollout,
giving **separate spatial and temporal heatmaps** overlaid on the input frames. Saves a
grid PNG per video under `visualizations/`. Use `--num_videos 8` for more.

---

## 5. If the runtime disconnects (it will, eventually)

Do nothing special. Just re-open the Studio and **run the exact same command again**:

- Mid-preprocessing? → re-run `python -m src.preprocess --dataset <name>`; it skips done videos.
- Mid-training? → re-run `python -m src.train`; it resumes from `last.pt`.

That's the whole recovery procedure. This is why the project is built the way it is.

---

## 6. Common problems & quick fixes

**"CUDA out of memory" during training.**
Edit `config.py`: set `BATCH_SIZE = 4` (and optionally `GRAD_ACCUM_STEPS = 4` to keep the
effective batch the same). Re-run `python -m src.train`.

**A dataset split is empty / very few videos in the manifest.**
You probably haven't preprocessed that dataset yet, or (for FF++) too many files were
corrupt. Either preprocess more, or swap the combo: make a *clean* dataset the cross-test
instead. Example — train on DFDC + FF++, cross-test on Celeb-DF:
```python
TRAIN_DATASETS = ["dfdc", "ff"]
CROSS_TEST_DATASET = "celebdf"
```
Then re-run `build_manifest` → `train` → `evaluate`.

**Preprocessing is slow.**
Make sure you're on a **GPU** machine for steps 3–5 (`MTCNN_DEVICE = "cuda"` in config,
which is the default; it auto-falls back to CPU if no GPU is present).

**I just want a fast smoke test first.**
In `config.py` set `MAX_VIDEOS_PER_DATASET = 400`, run the whole pipeline quickly to
confirm everything works end-to-end, then set it back to `0` (no cap) for the real run.

**No pretrained backbone downloaded (no internet on the node).**
The Xception entry-flow backbone tries to load ImageNet weights via `timm`; if that fails
it falls back to a random-init backbone automatically so the code still runs (accuracy
will be lower — prefer letting it download once).

---

## 7. Tuning for higher cross-dataset accuracy

In rough order of impact:
1. **Preprocess more clean data.** Set `MAX_VIDEOS_PER_DATASET = 0` and let both clean
   datasets fully process. More real/fake variety generalizes better.
2. **Train longer.** Raise `EPOCHS` (the paper goes up to 100). Watch validation AUROC in
   `logs/` and keep the `best.pt` it selects.
3. **Keep the pretrained backbone** (`USE_PRETRAINED_BACKBONE = True`).
4. **Keep class balancing on** (`BALANCE_CLASSES = True`).
5. **Sample more windows per video** at eval (`SEQUENCES_PER_VIDEO_EVAL`) for a steadier
   video-level score.
6. Try each train/cross-test combination — some pairs transfer better than others; report
   whichever the held-out AUROC likes best.

---

## 8. File-by-file map

```
istvt_lightning/
├── config.py                  # the ONE file you edit (paths + hyperparameters)
├── requirements.txt
├── README.md                  # this file
├── scripts/
│   └── 00_check_datasets.py   # CPU sanity check before you spend any GPU
└── src/
    ├── xception_backbone.py   # Xception entry flow -> 19x19x728 features
    ├── model.py               # ISTVT: decomposed spatial-temporal transformer + self-subtract
    ├── preprocess.py          # MTCNN face crop (nose-centered 1.25x, 300x300), resumable
    ├── build_manifest.py      # face crops -> manifest.csv, video-level train/val/test split
    ├── dataset.py             # 6-frame sequence loader, [-1,1] normalize, balanced sampler
    ├── engine.py              # train/eval loops, AMP, warmup+cosine LR, video-level metrics
    ├── train.py               # entry point; auto-resume from last.pt, saves best.pt
    ├── evaluate.py            # cross-dataset video-level AUROC + accuracy
    └── visualize.py           # interpretability: spatial & temporal attention heatmaps
```

### How the model matches the paper (quick reference)
- **Backbone:** Xception **entry flow** only → 300×300×3 turns into a **19×19×728** map.
- **Tokens:** 361 patches/frame + 1 spatial cls = **362 spatial tokens**; **T=6** frames
  + 1 temporal cls = **7 temporal tokens**; learnable position embeddings.
- **Transformer:** **M=12** *decomposed* spatial-temporal blocks, **N=8** heads, dim 728.
  Decomposing attention cuts cost from O(T²H²W²) to O(T²+H²W²).
- **Self-subtract (Eq. 3):** inside temporal attention, Q/K come from frame-difference
  residuals while V stays from the original features — this is what makes the model focus
  on **inter-frame inconsistencies**, the tell-tale of deepfakes.
- **Head:** the spatial-temporal class token → LayerNorm → Linear → 1 logit, trained with
  `BCEWithLogitsLoss`.
- **Evaluation:** scores are aggregated to **video level** (averaged), AUROC for
  cross-dataset — exactly the paper's protocol.

---

Built to run end-to-end on Lightning AI, to survive disconnects, and to be re-run as many
times as you need. Start at Step 1 and work down the table.
