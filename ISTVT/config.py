"""
config.py
=========
Single place to edit ALL paths and hyper-parameters for the ISTVT project.

You normally only need to edit the four RAW_* paths in the DATASETS section to
point at where you uploaded the datasets on Lightning AI. Everything else has
sensible defaults that reproduce the paper (300x300 faces, 6-frame sequences,
M=12 transformer blocks, N=8 heads).

Read the comments next to each value before changing it.
"""

import os

# ---------------------------------------------------------------------------
# 0. PROJECT ROOT  (where preprocessed faces, manifests and checkpoints live)
# ---------------------------------------------------------------------------
# On Lightning AI your Studio home is persistent. Keep everything under one
# folder so a runtime disconnect never loses your work.
PROJECT_ROOT = os.environ.get("ISTVT_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Derived working folders (created automatically, do not need editing).
FACES_DIR      = os.path.join(PROJECT_ROOT, "data", "faces")       # extracted face crops
MANIFEST_DIR   = os.path.join(PROJECT_ROOT, "data", "manifests")   # csv manifests
CKPT_DIR       = os.path.join(PROJECT_ROOT, "checkpoints")         # model checkpoints
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")                # training logs
VIS_DIR        = os.path.join(PROJECT_ROOT, "visualizations")      # interpretability outputs

for _d in (FACES_DIR, MANIFEST_DIR, CKPT_DIR, LOG_DIR, VIS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATASETS  --  EDIT THESE PATHS to where you uploaded the raw video data.
# ---------------------------------------------------------------------------
# Each entry is the folder that directly contains the dataset's videos / subfolders
# exactly as you downloaded them.
#
# Celeb-DF (v2) folder must contain: Celeb-real/  Celeb-synthesis/  YouTube-real/
#                                    List_of_testing_videos.txt
# DFDC sample folder must contain:   train_sample_videos/  (with metadata.json inside)
# FaceForensics++ folder must contain the manipulation subfolders, e.g.
#                                    DeepFakeDetection/  original/  Deepfakes/ ...
RAW_PATHS = {
    "celebdf": os.environ.get("CELEBDF_DIR", "/teamspace/studios/this_studio/datasets/celeb-df"),
    "dfdc":    os.environ.get("DFDC_DIR",    "/teamspace/studios/this_studio/datasets/dfdc"),
    "ff":      os.environ.get("FF_DIR",      "/teamspace/studios/this_studio/datasets/FaceForensics++_C23"),
}

# Which datasets are pooled for TRAINING, and which single one is held out as the
# UNSEEN cross-dataset test. This is the core of your "2 train + 1 cross-test" plan.
#
# DEFAULT (recommended because FF++ is partly corrupt -> use the 2 clean datasets
# for training and the corrupt one only for evaluation, where missing videos hurt
# less):
TRAIN_DATASETS = ["celebdf", "dfdc"]   # pooled together for training + in-domain val
CROSS_TEST_DATASET = "ff"              # completely unseen during training
#
# Alternative combos (just swap the two lines above):
#   TRAIN_DATASETS = ["celebdf", "ff"]; CROSS_TEST_DATASET = "dfdc"
#   TRAIN_DATASETS = ["dfdc", "ff"];    CROSS_TEST_DATASET = "celebdf"

# ---------------------------------------------------------------------------
# 2. PREPROCESSING  (face extraction)
# ---------------------------------------------------------------------------
FACE_SIZE          = 300     # paper: faces resized to 300x300
NUM_CLIPS          = 4       # short consecutive runs sampled across each video
CLIP_LEN           = 8       # frames per clip (>= SEQ_LEN). 4*8 = 32 faces/video
MTCNN_DEVICE       = "cuda"  # "cuda" (fast, recommended) or "cpu" (free, slow)
MTCNN_BATCH        = 8       # frames detected per MTCNN call
BOX_SCALE          = 1.25    # paper: 1.25 x max(face_w, face_h), centered on nose
MIN_FACE_PROB      = 0.90    # discard low-confidence detections
JPEG_QUALITY       = 95
# To keep a first run cheap you can cap how many videos per dataset get processed.
# 0 means "no cap" (process everything). Set e.g. 400 for a quick smoke test.
MAX_VIDEOS_PER_DATASET = 1500

# ---------------------------------------------------------------------------
# 3. MODEL  (ISTVT -- matches the paper)
# ---------------------------------------------------------------------------
SEQ_LEN            = 6       # paper: 6 continuous frames per sequence (T)
FEATURE_GRID       = 19      # 300x300 -> Xception entry flow -> 19x19 feature map
EMBED_DIM          = 728     # Xception entry-flow channel count (C)
DEPTH              = 12      # paper: M = 12 transformer blocks
NUM_HEADS          = 8       # paper: N = 8 heads
DIM_HEAD           = 64
MLP_SCALE          = 4
DROPOUT            = 0.0
USE_PRETRAINED_BACKBONE = True  # load ImageNet-pretrained Xception (recommended)

# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------
EPOCHS             = 20       # paper trains up to 100; 20 is a strong, cheaper default
BATCH_SIZE         = 8        # sequences per step. Lower to 4 if you hit OOM.
GRAD_ACCUM_STEPS   = 2        # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_WORKERS        = 8
OPTIMIZER          = "adamw"  # "adamw" (stable default) or "sgd" (paper)
LEARNING_RATE      = 1e-4     # adamw default; for sgd the paper uses 5e-4 + warmup
WEIGHT_DECAY       = 0.05
WARMUP_EPOCHS      = 2
USE_AMP            = True     # mixed precision -> ~2x faster, ~half memory
VAL_SPLIT          = 0.10     # fraction of training videos held out for in-domain val
BALANCE_CLASSES    = True     # weighted sampling so real/fake are balanced each epoch
SEQUENCES_PER_VIDEO_TRAIN = 4 # random windows drawn per video each epoch
SEQUENCES_PER_VIDEO_EVAL  = 6 # windows averaged for a video-level score at eval
SEED               = 1337
GRAD_CLIP          = 1.0

# ---------------------------------------------------------------------------
# 5. CHECKPOINTING  (so a disconnect never costs you an epoch)
# ---------------------------------------------------------------------------
RESUME             = True     # auto-resume from last.pt if it exists
SAVE_EVERY_EPOCH   = True
RUN_NAME           = "istvt_" + "_".join(TRAIN_DATASETS) + "__x_" + CROSS_TEST_DATASET


def describe():
    """Print the active configuration (handy as the first cell of any run)."""
    print("=" * 70)
    print("ISTVT configuration")
    print("=" * 70)
    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    print(f"TRAIN_DATASETS      : {TRAIN_DATASETS}")
    print(f"CROSS_TEST_DATASET  : {CROSS_TEST_DATASET}")
    for name, p in RAW_PATHS.items():
        exists = "OK" if os.path.isdir(p) else "MISSING"
        print(f"  raw[{name:8s}]     : {p}   [{exists}]")
    print(f"SEQ_LEN / DEPTH/HEADS: {SEQ_LEN} / {DEPTH} / {NUM_HEADS}")
    print(f"BATCH/ACCUM/EPOCHS  : {BATCH_SIZE} / {GRAD_ACCUM_STEPS} / {EPOCHS}")
    print(f"RUN_NAME            : {RUN_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    describe()
