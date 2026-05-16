"""
train.py — 5-fold cross-validation training for DigitNet.

Usage (on Kaggle):
    python src/train.py
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import (
    SEED, DEVICE, TRAIN_CSV, TRAIN_AUDIO, OUTPUT_DIR,
    SR, N_FFT, HOP, N_MELS, BATCH_SIZE, EPOCHS, LR, N_FOLDS,
)
from utils import set_seed, load_wav
from dataset import DigitDataset
from model import DigitNet


# ── Mel transform on GPU ──────────────────────────────────────
mel_gpu = MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS).to(DEVICE)
db_gpu  = AmplitudeToDB().to(DEVICE)


def wav_to_spec_gpu(wav_batch: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, T) waveform batch to normalized Mel-spectrogram on GPU."""
    spec = db_gpu(mel_gpu(wav_batch))
    mean = spec.mean(dim=[2, 3], keepdim=True)
    std  = spec.std(dim=[2, 3],  keepdim=True)
    return (spec - mean) / (std + 1e-6)


def spec_augment_gpu(spec: torch.Tensor) -> torch.Tensor:
    """SpecAugment: random time and frequency masking."""
    B, C, F, T = spec.shape
    for _ in range(2):
        if random.random() < 0.5:
            t  = random.randint(5, 20)
            t0 = random.randint(0, max(1, T - t))
            spec[:, :, :, t0:t0 + t] = 0
    for _ in range(2):
        if random.random() < 0.5:
            f  = random.randint(4, 15)
            f0 = random.randint(0, max(1, F - f))
            spec[:, :, f0:f0 + f, :] = 0
    return spec


def train_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    for wav, y in loader:
        wav = wav.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE,   non_blocking=True)

        with torch.no_grad():
            spec = wav_to_spec_gpu(wav)
        spec = spec_augment_gpu(spec)

        # Mixup augmentation
        if random.random() < 0.5:
            lam  = np.random.beta(0.3, 0.3)
            idx  = torch.randperm(spec.size(0))
            spec = lam * spec + (1 - lam) * spec[idx]
            y_b  = y[idx]
            optimizer.zero_grad()
            pred = model(spec)
            loss = lam * loss_fn(pred, y) + (1 - lam) * loss_fn(pred, y_b)
        else:
            optimizer.zero_grad()
            loss = loss_fn(model(spec), y)

        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for wav, y in loader:
            wav  = wav.to(DEVICE, non_blocking=True)
            spec = wav_to_spec_gpu(wav)
            preds.append(model(spec).argmax(1).cpu())
            targets.append(y)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    acc     = (preds == targets).mean()
    f1      = f1_score(targets, preds, average="macro")
    return acc, f1


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load all training audio into RAM ──────────────────────
    print("Loading all training audio into RAM...")
    df_full    = pd.read_csv(TRAIN_CSV)
    all_wavs   = {}
    all_labels = {}

    for _, row in tqdm(df_full.iterrows(), total=len(df_full)):
        path               = os.path.join(TRAIN_AUDIO, f"{row.id}.wav")
        all_wavs[row.id]   = load_wav(path)
        all_labels[row.id] = int(row.label)

    gc.collect()
    print(f"Loaded {len(all_wavs)} samples.")

    # ── 5-Fold Cross Validation ───────────────────────────────
    ids, lbls = df_full["id"].values, df_full["label"].values
    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(ids, lbls)):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*55}")

        train_ds = DigitDataset(ids[train_idx], all_wavs, all_labels, train=True)
        val_ds   = DigitDataset(ids[val_idx],   all_wavs, all_labels, train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model     = DigitNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )
        loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_f1   = 0.0
        save_path = os.path.join(OUTPUT_DIR, f"best_model_fold{fold}.pt")

        for epoch in range(EPOCHS):
            loss           = train_epoch(model, train_loader, optimizer, loss_fn)
            val_acc, val_f1 = evaluate(model, val_loader)
            scheduler.step()
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss {loss:.4f} | "
                  f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), save_path)

        print(f"\nFold {fold} best Val F1: {best_f1:.4f}")
        fold_f1s.append(best_f1)

    print(f"\nAll folds done! Mean CV F1: {np.mean(fold_f1s):.4f}")


if __name__ == "__main__":
    main()
