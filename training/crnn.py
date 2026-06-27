"""CRNN recognizer, CTC label converter, and dataset for OCR fine-tuning.

The architecture mirrors EasyOCR's "standard" recognizer closely enough that
the trained state dict can be adapted to EasyOCR's custom-network format:
a VGG-style feature extractor, a 2-layer BiLSTM, and a linear CTC head.

Torch is imported at module top because this module is only ever imported by
the training script, which already guarantees torch is present.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Fixed input geometry for the recognizer.
IMG_HEIGHT = 32
IMG_WIDTH = 256


class CTCLabelConverter:
    """Map between strings and integer label sequences for CTC (blank=0)."""

    def __init__(self, charset: list[str]):
        # index 0 reserved for the CTC blank.
        self.charset = charset
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(charset)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        # Collapse repeats and drop blanks (standard greedy CTC decode).
        out = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:
                out.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(out)


class RecognitionDataset(Dataset):
    def __init__(self, samples, converter: CTCLabelConverter):
        self.samples = samples
        self.converter = converter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        raw = np.fromfile(s.image_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        labels = torch.tensor(self.converter.encode(s.text), dtype=torch.long)
        return tensor, labels


def collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.cat([b[1] for b in batch]) if batch else torch.tensor([])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return images, targets, lengths


class CRNN(nn.Module):
    """VGG features -> BiLSTM -> CTC logits."""

    def __init__(self, num_classes: int, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),     # 16x128
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),   # 8x64
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                     # 4x64
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                     # 2x64
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True), # 1x63
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feats = self.cnn(x)                 # (N, C, 1, W')
        feats = feats.squeeze(2)            # (N, C, W')
        feats = feats.permute(2, 0, 1)      # (W'=T, N, C)
        rnn_out, _ = self.rnn(feats)        # (T, N, 2*hidden)
        logits = self.fc(rnn_out)           # (T, N, num_classes)
        return logits


def load_recognizer(weights_path: str | Path, device: str = "cpu"):
    """Load a trained recognizer for inference/benchmarking."""
    ckpt = torch.load(str(weights_path), map_location=device)
    charset = ckpt["charset"]
    model = CRNN(num_classes=len(charset) + 1)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, CTCLabelConverter(charset)
