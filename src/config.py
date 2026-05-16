import os
import torch

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = "/kaggle/input/competitions/digitrecognition-ee708"
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TRAIN_AUDIO = os.path.join(BASE_DIR, "train_audio", "train_audio")
TEST_AUDIO  = os.path.join(BASE_DIR, "test_audio", "test_audio")
OUTPUT_DIR  = "/kaggle/working"
TEST_CSV    = os.path.join(OUTPUT_DIR, "test.csv")

# ── Audio ─────────────────────────────────────────────────────
SR          = 16000
MAX_SAMPLES = 16000      # 1 second of audio
N_MELS      = 64
N_FFT       = 400
HOP         = 160

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 1e-3
N_FOLDS     = 5

# ── Test-Time Augmentation ────────────────────────────────────
TTA_SHIFTS  = [0, 800, -800, 400, -400, 1200, -1200, 200, -200, 600, -600]

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
