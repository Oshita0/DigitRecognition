# Spoken Digit Recognition

Audio classification of spoken digits (0–9) from `.wav` clips using a custom CNN with Inception-style blocks.

**Results:** Public F1: `0.9964` | Private F1: `0.9960` (Macro F1, 10-class)

---

## Problem

Given short `.wav` audio clips of spoken digits, predict the digit (0–9). The test set contains **unseen speakers** and **noisy/augmented recordings**, making speaker-independent generalization critical.

**Metric:** Macro F1-score across all 10 digit classes.

---

## Approach

### Feature Extraction
- Raw waveforms resampled to **16 kHz**, padded/cropped to **1 second**
- **Mel-spectrogram** (64 mels, FFT=400, hop=160) computed on-the-fly on GPU
- Per-sample **z-score normalization** of spectrograms

### Model — `DigitNet`
A custom CNN built around multi-scale Inception blocks:

```
Input (1, 64, T)
    └── Stem: Conv(64) → MaxPool → Conv(128) → MaxPool
    └── InceptionBlock × 2  [256 ch]
    └── MaxPool
    └── InceptionBlock × 2  [256 ch]
    └── Global Average Pool
    └── MLP: 256 → 128 → 10
```

Each `InceptionBlock` fuses 1×1, 3×3, and 5×5 convolution branches in parallel (inspired by GoogLeNet), capturing multi-scale spectro-temporal patterns.

### Training
| Setting | Value |
|---|---|
| Folds | 5-fold Stratified CV |
| Epochs | 50 per fold |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | CosineAnnealingWarmRestarts |
| Loss | CrossEntropy (label smoothing=0.1) |
| Batch size | 128 |

### Augmentation
**Waveform-level (DigitDataset):**
- Additive Gaussian noise
- Random time-shift (±3200 samples)
- Random amplitude scaling (0.7×–1.3×)

**Spectrogram-level (GPU, SpecAugment):**
- 2× random time masking (5–20 frames)
- 2× random frequency masking (4–15 bins)

**Mixup** (50% probability per batch, α=0.3)

### Inference — Test-Time Augmentation (TTA)
Predictions averaged across **5 fold models × 11 time-shifts** = 55 forward passes per sample.

TTA shifts: `[0, ±200, ±400, ±600, ±800, ±1200]` samples

---

## Repository Structure

```
Digit_Recognition/
├── src/
│   ├── config.py       # All hyperparameters and paths
│   ├── utils.py        # Seed fixing, audio loading helpers
│   ├── dataset.py      # DigitDataset, TestDataset
│   ├── model.py        # InceptionBlock, DigitNet
│   ├── train.py        # 5-fold CV training loop
│   └── predict.py      # TTA inference + submission CSV
├── notebooks/          # Original Kaggle notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Usage

> **Note:** This code is designed to run on Kaggle with GPU. Update the paths in `src/config.py` to match your environment.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python src/train.py
```
Saves `best_model_fold{0..4}.pt` to `OUTPUT_DIR`.

### 3. Predict
```bash
python src/predict.py
```
Saves `submission.csv` to `OUTPUT_DIR`.

---

## Key Design Decisions

- **All audio pre-loaded into RAM** before training to avoid repeated disk I/O bottleneck
- **Mel transform on GPU** to keep the CPU→GPU pipeline saturated
- **SpecAugment applied post-transfer** (on GPU) for speed
- **Stratified K-Fold** ensures balanced digit class distribution across folds
- **Label smoothing** reduces overconfidence on clean training clips
