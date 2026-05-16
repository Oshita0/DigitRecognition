import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio

from config import SEED, SR, MAX_SAMPLES


def set_seed(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_crop(wav: torch.Tensor) -> torch.Tensor:
    """Ensure waveform is exactly MAX_SAMPLES long."""
    if wav.shape[-1] > MAX_SAMPLES:
        return wav[..., :MAX_SAMPLES]
    return F.pad(wav, (0, MAX_SAMPLES - wav.shape[-1]))


def load_wav(path: str) -> torch.Tensor:
    """
    Load a .wav file, convert to mono, resample to SR if needed,
    and pad/crop to MAX_SAMPLES.
    Returns a (1, MAX_SAMPLES) float32 tensor.
    """
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data).unsqueeze(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return pad_or_crop(wav)
