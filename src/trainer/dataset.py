"""
trainer/dataset.py - VSL Dataset + DataLoader builder
======================================================
    from trainer.dataset import VSLDataset, build_dataloaders, compute_split_counts
"""

import os
import math
import json
import numpy as np
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split as sk_split

from vsl.config import cfg as vsl_cfg


class VSLDataset(Dataset):
    """Đọc file .npy từ data/processed/<label>/*.npy → shape (seq_len, feat_dim)"""

    def __init__(self, data_dir: str, label_map: dict, augment: bool = False):
        self.samples = []
        self.augment = augment
        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir):
                print(f"  CANH BAO: Khong tim thay: {label_dir}")
                continue
            for fp in sorted(Path(label_dir).glob('*.npy')):
                self.samples.append((str(fp), label_idx))
        print(f"  Dataset: {len(self.samples)} samples, {len(label_map)} classes")

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
        # Gaussian noise
        if np.random.rand() < 0.5:
            data = data + np.random.normal(0, 0.002, data.shape).astype(np.float32)

        # Random temporal crop
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
    """Đếm số mẫu mỗi class trong từng split — dùng cho biểu đồ phân bổ."""
    idx2label = {v: k for k, v in label_map.items()}
    counts    = {name: {'train': 0, 'val': 0, 'test': 0}
                 for name in label_map}

    def _count(ds, split_name):
        for _, label_idx in ds:
            li   = label_idx.item() if isinstance(label_idx, torch.Tensor) \
                   else label_idx
            name = idx2label.get(li, str(li))
            if name in counts:
                counts[name][split_name] += 1

    _count(train_ds, 'train')
    _count(val_ds,   'val')
    _count(test_ds,  'test')
    return counts


def build_dataloaders(data_dir: str, label_map: dict, cfg) -> tuple:
    """
    Stratified split → đảm bảo mỗi split có đủ tất cả class.
    Fallback về random split nếu có class < 3 mẫu.

    Trả về: (train_loader, val_loader, test_loader, split_counts)
    """
    full_ds = VSLDataset(data_dir, label_map, augment=False)
    if len(full_ds) == 0:
        raise ValueError("Dataset trong! Kiem tra thu muc data/processed/")

    all_indices  = list(range(len(full_ds)))
    all_labels   = [full_ds.samples[i][1] for i in all_indices]
    class_counts = Counter(all_labels)
    min_count    = min(class_counts.values())
    test_ratio   = 1.0 - cfg.TRAIN_RATIO - cfg.VAL_RATIO

    if min_count < 3:
        print(f"  CANH BAO: Co class chi co {min_count} mau → dung random split")
        np.random.seed(42)
        np.random.shuffle(all_indices)
        n         = len(all_indices)
        n_test    = max(1, int(n * test_ratio))
        n_val     = max(1, int(n * cfg.VAL_RATIO))
        test_idx  = all_indices[:n_test]
        val_idx   = all_indices[n_test:n_test + n_val]
        train_idx = all_indices[n_test + n_val:]
    else:
        train_idx, temp_idx, _, temp_labels = sk_split(
            all_indices, all_labels,
            test_size=(test_ratio + cfg.VAL_RATIO),
            stratify=all_labels, random_state=42,
        )
        val_idx, test_idx = sk_split(
            temp_idx, test_size=0.5,
            stratify=temp_labels, random_state=42,
        )

    print(f"  Split: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    train_aug_ds = VSLDataset(data_dir, label_map, augment=True)
    train_ds = Subset(train_aug_ds, train_idx)
    val_ds   = Subset(VSLDataset(data_dir, label_map, augment=False), val_idx)
    test_ds  = Subset(VSLDataset(data_dir, label_map, augment=False), test_idx)

    kw = dict(num_workers=0, pin_memory=(cfg.DEVICE == 'cuda'))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)

    split_counts = compute_split_counts(train_ds, val_ds, test_ds, label_map)
    return train_loader, val_loader, test_loader, split_counts