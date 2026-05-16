import random
import torch
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    """
    Dataset for training/validation.
    Reads pre-loaded waveforms from in-memory dicts and applies
    on-the-fly waveform augmentation during training.
    """

    def __init__(self, ids, all_wavs: dict, all_labels: dict, train: bool = True):
        self.ids        = ids
        self.all_wavs   = all_wavs
        self.all_labels = all_labels
        self.train      = train

    def __len__(self):
        return len(self.ids)

    def augment_wave(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.clone()
        # Additive Gaussian noise
        if random.random() < 0.6:
            wav = wav + random.uniform(0.001, 0.015) * torch.randn_like(wav)
        # Random time-shift
        if random.random() < 0.6:
            shift = random.randint(-3200, 3200)
            wav   = torch.roll(wav, shifts=shift, dims=-1)
            if shift > 0:
                wav[..., :shift] = 0
            else:
                wav[..., shift:] = 0
        # Amplitude scaling
        if random.random() < 0.5:
            wav = wav * random.uniform(0.7, 1.3)
        return wav

    def __getitem__(self, idx):
        aid = self.ids[idx]
        wav = self.all_wavs[aid].clone()
        if self.train:
            wav = self.augment_wave(wav)
        return wav, self.all_labels[aid]


class TestDataset(Dataset):
    """
    Dataset for inference with optional time-shift for TTA.
    """

    def __init__(self, ids, all_test_wavs: dict, shift: int = 0):
        self.ids           = ids
        self.all_test_wavs = all_test_wavs
        self.shift         = shift

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        aid = self.ids[idx]
        wav = self.all_test_wavs[aid].clone()
        if self.shift != 0:
            wav = torch.roll(wav, shifts=self.shift, dims=-1)
            if self.shift > 0:
                wav[..., :self.shift] = 0
            else:
                wav[..., self.shift:] = 0
        return wav
