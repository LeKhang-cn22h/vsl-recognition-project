"""
trainer/dataset.py - VSL Dataset + DataLoader builder
======================================================
    from trainer.dataset import VSLDataset, build_dataloaders, compute_split_counts

Cấu trúc folder:
    data/processed/
    ├── train/<label>/*.npy
    ├── val/<label>/*.npy
    └── test/<label>/*.npy
"""

import os
import math
import json
import numpy as np
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from vsl.config import cfg as vsl_cfg


class VSLDataset(Dataset):
    """
    Đọc file .npy từ data/processed/<split>/<label>/*.npy
    → shape (seq_len, feat_dim)
    """

    def __init__(self, data_dir: str, label_map: dict,
                 split: str = 'train', augment: bool = False):
        """
        data_dir : thư mục gốc, ví dụ 'data/processed'
        split    : 'train' | 'val' | 'test'
        augment  : runtime augmentation nhẹ (chỉ dùng cho train)
        """
        self.samples = []
        self.augment = augment
        split_dir    = os.path.join(data_dir, split)

        if not os.path.isdir(split_dir):
            print(f"  CANH BAO: Khong tim thay split dir: {split_dir}")
            return

        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                print(f"  CANH BAO: [{split}] Khong tim thay: {label_dir}")
                continue
            npy_files = sorted(Path(label_dir).glob('*.npy'))
            if not npy_files:
                print(f"  CANH BAO: [{split}/{label_name}] Khong co file .npy")
                continue
            for fp in npy_files:
                self.samples.append((str(fp), label_idx))

        print(f"  [{split:5s}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)
        if self.augment:
            data = self._runtime_aug(data)
        return torch.from_numpy(data), label

    def _runtime_aug(self, data: np.ndarray) -> np.ndarray:
        """Runtime augmentation nhẹ: noise + random temporal crop."""
        if np.random.rand() < 0.5:
            data = data + np.random.normal(0, 0.002, data.shape).astype(np.float32)

        if np.random.rand() < 0.3:
            T     = data.shape[0]
            start = np.random.randint(0, max(1, T // 10))
            end   = np.random.randint(min(T-1, T - T//10), T)
            crop  = data[start:end]
            if len(crop) >= 2:
                idx_f    = np.linspace(0, len(crop)-1, T)
                new_data = np.zeros_like(data)
                for i, fi in enumerate(idx_f):
                    lo = int(math.floor(fi))
                    hi = min(int(math.ceil(fi)), len(crop)-1)
                    w  = fi - lo
                    new_data[i] = crop[lo] * (1-w) + crop[hi] * w
                data = new_data
        return data


def compute_split_counts(train_ds, val_ds, test_ds, label_map: dict) -> dict:
    """Đếm số mẫu mỗi class trong từng split."""
    idx2label = {v: k for k, v in label_map.items()}
    counts    = {name: {'train': 0, 'val': 0, 'test': 0}
                 for name in label_map}

    def _count(ds, split_name):
        for _, label_idx in ds.samples:
            name = idx2label.get(label_idx, str(label_idx))
            if name in counts:
                counts[name][split_name] += 1

    _count(train_ds, 'train')
    _count(val_ds,   'val')
    _count(test_ds,  'test')
    return counts


def build_dataloaders(data_dir: str, label_map: dict, cfg) -> tuple:
    """
    Đọc train/val/test từ split folder riêng biệt.
    Trả về: (train_loader, val_loader, test_loader, split_counts)
    """
    train_ds = VSLDataset(data_dir, label_map, split='train', augment=True)
    val_ds   = VSLDataset(data_dir, label_map, split='val',   augment=False)
    test_ds  = VSLDataset(data_dir, label_map, split='test',  augment=False)

    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"  Tong: {total} samples "
          f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")

    if len(train_ds) == 0:
        raise ValueError(
            "Train dataset trong!\n"
            f"  Kiem tra thu muc: {os.path.join(data_dir, 'train')}\n"
            "  Chay video_to_npy.py truoc de tao file .npy"
        )

    kw = dict(num_workers=0, pin_memory=(cfg.DEVICE == 'cuda'))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)

    split_counts = compute_split_counts(train_ds, val_ds, test_ds, label_map)
    return train_loader, val_loader, test_loader, split_counts