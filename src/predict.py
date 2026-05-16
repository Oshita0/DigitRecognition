"""
predict.py — TTA inference using all saved fold models.

Usage (on Kaggle after training):
    python src/predict.py
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm

from config import (
    DEVICE, OUTPUT_DIR, TEST_AUDIO, TEST_CSV,
    SR, N_FFT, HOP, N_MELS, BATCH_SIZE, N_FOLDS, TTA_SHIFTS,
)
from utils import load_wav
from dataset import TestDataset
from model import DigitNet


# ── Mel transform on GPU ──────────────────────────────────────
mel_gpu = MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS).to(DEVICE)
db_gpu  = AmplitudeToDB().to(DEVICE)


def wav_to_spec_gpu(wav_batch: torch.Tensor) -> torch.Tensor:
    spec = db_gpu(mel_gpu(wav_batch))
    mean = spec.mean(dim=[2, 3], keepdim=True)
    std  = spec.std(dim=[2, 3],  keepdim=True)
    return (spec - mean) / (std + 1e-6)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Build test.csv if missing ─────────────────────────────
    if not os.path.exists(TEST_CSV):
        files   = sorted([f for f in os.listdir(TEST_AUDIO) if f.endswith(".wav")])
        ids     = [os.path.splitext(f)[0] for f in files]
        pd.DataFrame({"id": ids}).to_csv(TEST_CSV, index=False)
        print(f"Built test.csv with {len(ids)} entries.")

    # ── Load all test audio into RAM ──────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loading {len(test_df)} test clips into RAM...")
    all_test_wavs = {}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        path = os.path.join(TEST_AUDIO, f"{row.id}.wav")
        all_test_wavs[row.id] = load_wav(path)
    gc.collect()

    # ── Load fold models ──────────────────────────────────────
    models = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(OUTPUT_DIR, f"best_model_fold{fold}.pt")
        if not os.path.exists(ckpt):
            print(f"WARNING: fold {fold} checkpoint not found, skipping.")
            continue
        m = DigitNet().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} fold models.")

    # ── TTA Inference ─────────────────────────────────────────
    test_ids  = test_df["id"].values
    all_probs = []

    for shift in TTA_SHIFTS:
        print(f"TTA shift={shift:+d} ...")
        ds     = TestDataset(test_ids, all_test_wavs, shift=shift)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        for model in models:
            probs = []
            with torch.no_grad():
                for wav in loader:
                    wav  = wav.to(DEVICE, non_blocking=True)
                    spec = wav_to_spec_gpu(wav)
                    prob = torch.softmax(model(spec), dim=1).cpu().numpy()
                    probs.append(prob)
            all_probs.append(np.concatenate(probs, axis=0))

    # ── Aggregate and save ────────────────────────────────────
    avg_probs  = np.mean(all_probs, axis=0)
    preds      = avg_probs.argmax(axis=1)

    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    pd.DataFrame({"id": test_ids, "label": preds}).to_csv(sub_path, index=False)

    print(f"\nSaved → {sub_path}")
    print(f"Total predictions : {len(preds)}")
    print("Label distribution:")
    print(pd.Series(preds).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
