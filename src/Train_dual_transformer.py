"""
DUAL TRANSFORMER TRAINER - VSL Sign Language Recognition
Kiến trúc: Spatial Transformer + Temporal Transformer
Tích hợp đầy đủ biểu đồ cho báo cáo nghiên cứu khoa học.
"""

import os
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import classification_report

# Import Visualizer (cùng thư mục)
from visualize_training import Visualizer


# ═══════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════

class Config:
    # ── Data ──
    DATA_DIR        = 'data/processed'
    LABEL_MAP_PATH  = 'data/processed/label_map.json'
    SEQ_LEN         = 30
    FEAT_DIM        = 339 

    # ── Feature group boundaries ──
    POSE_START,    POSE_END    = 0,   75
    FACE_START,    FACE_END    = 75,  165
    HAND_START,    HAND_END    = 165, 291
    BLEND_START,   BLEND_END   = 291, 308
    INTERACT_START,INTERACT_END= 308, 339

    # ── Model ──
    D_MODEL           = 256
    SPATIAL_HEADS     = 8
    SPATIAL_LAYERS    = 3
    SPATIAL_FF_DIM    = 512
    SPATIAL_DROPOUT   = 0.1
    TEMPORAL_HEADS    = 8
    TEMPORAL_LAYERS   = 4
    TEMPORAL_FF_DIM   = 512
    TEMPORAL_DROPOUT  = 0.1
    CLASSIFIER_HIDDEN = 256
    DROPOUT_FINAL     = 0.3

    # ── Training ──
    EPOCHS       = 100
    BATCH_SIZE   = 32
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    TRAIN_RATIO  = 0.8
    VAL_RATIO    = 0.1
    PATIENCE     = 15
    GRAD_CLIP    = 1.0

    # ── Output ──
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR        = 'logs'
    CHART_DIR      = 'charts'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════

class VSLDataset(Dataset):
    def __init__(self, data_dir, label_map, augment=False):
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

    def _runtime_aug(self, data):
        if np.random.rand() < 0.5:
            data += np.random.normal(0, 0.002, data.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            T     = data.shape[0]
            start = np.random.randint(0, max(1, T // 10))
            end   = np.random.randint(min(T - 1, T - T // 10), T)
            crop  = data[start:end]
            if len(crop) >= 2:
                idx_f    = np.linspace(0, len(crop) - 1, T)
                new_data = np.zeros_like(data)
                for i, fi in enumerate(idx_f):
                    lo = int(math.floor(fi))
                    hi = min(int(math.ceil(fi)), len(crop) - 1)
                    w  = fi - lo
                    new_data[i] = crop[lo] * (1 - w) + crop[hi] * w
                data = new_data
        return data


# ═══════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════
# SPATIAL TRANSFORMER
# ═══════════════════════════════════════════════════════════

class SpatialTransformer(nn.Module):
    NUM_TOKENS = 6

    def __init__(self, feat_dim, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.d_model = d_model
        group_dims   = [
            cfg.POSE_END    - cfg.POSE_START,
            cfg.FACE_END    - cfg.FACE_START,
            21 * 3,
            21 * 3,
            cfg.BLEND_END   - cfg.BLEND_START,
            cfg.INTERACT_END - cfg.INTERACT_START,
        ]
        self.group_projs = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in group_dims
        ])
        self.token_embed = nn.Embedding(self.NUM_TOKENS, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _split_groups(self, x):
        return [
            x[:, :, cfg.POSE_START    : cfg.POSE_END],
            x[:, :, cfg.FACE_START    : cfg.FACE_END],
            x[:, :, cfg.HAND_START    : cfg.HAND_START + 63],
            x[:, :, cfg.HAND_START+63 : cfg.HAND_END],
            x[:, :, cfg.BLEND_START   : cfg.BLEND_END],
            x[:, :, cfg.INTERACT_START: cfg.INTERACT_END],
        ]

    def forward(self, x):
        B, T, _ = x.shape
        groups   = self._split_groups(x)
        toks     = []
        for i, (g, proj) in enumerate(zip(groups, self.group_projs)):
            tok = proj(g.reshape(B * T, -1)) + self.token_embed.weight[i]
            toks.append(tok.unsqueeze(1))
        tokens = torch.cat(toks, dim=1)
        cls    = self.cls_token.expand(B * T, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out    = self.transformer(tokens)
        return out[:, 0, :].reshape(B, T, self.d_model)


# ═══════════════════════════════════════════════════════════
# TEMPORAL TRANSFORMER
# ═══════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, ff_dim, dropout, seq_len):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1,
                                          dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        B   = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = self.pos_enc(torch.cat([cls, x], dim=1))
        out = self.transformer(x)
        return out[:, 0, :]


# ═══════════════════════════════════════════════════════════
# DUAL TRANSFORMER
# ═══════════════════════════════════════════════════════════

class DualTransformer(nn.Module):
    def __init__(self, feat_dim, seq_len, num_classes, cfg: Config):
        super().__init__()
        d = cfg.D_MODEL
        self.spatial  = SpatialTransformer(
            feat_dim, d, cfg.SPATIAL_HEADS, cfg.SPATIAL_LAYERS,
            cfg.SPATIAL_FF_DIM, cfg.SPATIAL_DROPOUT,
        )
        self.temporal = TemporalTransformer(
            d, cfg.TEMPORAL_HEADS, cfg.TEMPORAL_LAYERS,
            cfg.TEMPORAL_FF_DIM, cfg.TEMPORAL_DROPOUT, seq_len,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d * 2, cfg.CLASSIFIER_HIDDEN),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_FINAL),
            nn.Linear(cfg.CLASSIFIER_HIDDEN, cfg.CLASSIFIER_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_FINAL / 2),
            nn.Linear(cfg.CLASSIFIER_HIDDEN // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        spatial_seq  = self.spatial(x)
        temporal_out = self.temporal(spatial_seq)
        spatial_avg  = spatial_seq.mean(dim=1)
        fused = torch.cat([spatial_avg, temporal_out], dim=-1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 label_map, cfg, split_counts=None):
        self.model        = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.label_map    = label_map
        self.cfg          = cfg
        self.device       = cfg.DEVICE
        self.split_counts = split_counts

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        self.best_val_acc = 0.0
        self.patience_cnt = 0

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f'best_{ts}.pt')
        self.log_path  = os.path.join(cfg.LOG_DIR, f'history_{ts}.json')

        # ── Khởi tạo Visualizer ──
        self.viz = Visualizer(label_map, output_dir=cfg.CHART_DIR)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"\n  Model   : DualTransformer | Params: {n_params:,}")
        print(f"  Device  : {self.device}")
        print(f"  Train/Val/Test: "
              f"{len(train_loader.dataset)}/"
              f"{len(val_loader.dataset)}/"
              f"{len(test_loader.dataset)}")
        print(f"  Charts  : {cfg.CHART_DIR}/")
        print(f"  Ckpt    : {self.ckpt_path}\n")

    # ── 1 epoch ────────────────────────────────────────────
    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss   = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.GRAD_CLIP)
                    self.optimizer.step()
                total_loss += loss.item() * len(y)
                correct    += (logits.argmax(-1) == y).sum().item()
                total      += len(y)
        return total_loss / total, correct / total

    # ── Training loop ──────────────────────────────────────
    def train(self):
        print("="*60)
        print(" BAT DAU TRAINING ".center(60))
        print("="*60)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            tl, ta = self._run_epoch(self.train_loader, True)
            vl, va = self._run_epoch(self.val_loader,   False)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # ── Ghi nhận vào Visualizer ──
            self.viz.update(epoch, tl, vl, ta, va, lr)

            print(f"  Epoch {epoch:3d}/{self.cfg.EPOCHS} | "
                  f"Loss {tl:.4f}/{vl:.4f} | "
                  f"Acc {ta*100:5.1f}%/{va*100:5.1f}% | "
                  f"LR {lr:.2e}")

            if va > self.best_val_acc:
                self.best_val_acc = va
                self.patience_cnt = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_acc': va,
                    'label_map': self.label_map,
                }, self.ckpt_path)
                print(f"  ✓ Best saved (val={va*100:.1f}%)")
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= self.cfg.PATIENCE:
                    print(f"\n  Early stopping @ epoch {epoch}")
                    break

        # Lưu history JSON
        with open(self.log_path, 'w') as f:
            json.dump(self.viz.history, f, indent=2)
        print(f"\n  History saved: {self.log_path}")

    # ── Evaluate + xuất tất cả biểu đồ ────────────────────
    def evaluate_and_plot(self):
        print("\n" + "="*60)
        print(" EVALUATE + XUAT BIEU DO ".center(60))
        print("="*60)

        # Load best checkpoint
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])

        # Xuất toàn bộ 13 biểu đồ
        self.viz.plot_all(
            model        = self.model,
            test_loader  = self.test_loader,
            device       = self.device,
            cfg          = self.cfg,
            split_counts = self.split_counts,
        )

        # Classification report in terminal
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                preds = self.model(X.to(self.device)).argmax(-1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())

        idx2label = {v: k for k, v in self.label_map.items()}
        names     = [idx2label[i] for i in range(len(self.label_map))]
        test_acc  = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
        print(f"  Best Val Acc  : {self.best_val_acc*100:.2f}%")
        print("\n  Classification Report:")
        print(classification_report(all_labels, all_preds,
                                     target_names=names, zero_division=0))
        return test_acc


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def compute_split_counts(train_ds, val_ds, test_ds, label_map):
    """Đếm số mẫu mỗi class trong từng split (cho biểu đồ phân bổ)"""
    idx2label = {v: k for k, v in label_map.items()}
    counts    = {name: {'train': 0, 'val': 0, 'test': 0}
                 for name in label_map}

    def _count(ds, split_name):
        for _, label_idx in ds:
            li = label_idx.item() if isinstance(label_idx, torch.Tensor) \
                 else label_idx
            name = idx2label.get(li, str(li))
            if name in counts:
                counts[name][split_name] += 1

    _count(train_ds, 'train')
    _count(val_ds,   'val')
    _count(test_ds,  'test')
    return counts


def load_label_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay {path}\nHay chay video_to_npy.py truoc!")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_dataloaders(data_dir, label_map, cfg):
    full_ds = VSLDataset(data_dir, label_map, augment=False)
    if len(full_ds) == 0:
        raise ValueError("Dataset trong! Kiem tra thu muc data/processed/")

    n       = len(full_ds)
    n_val   = max(1, int(n * cfg.VAL_RATIO))
    n_test  = max(1, n - int(n * cfg.TRAIN_RATIO) - n_val)
    n_train = n - n_val - n_test

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=g)

    # Augmentation chỉ cho train
    train_ds.dataset = VSLDataset(data_dir, label_map, augment=True)

    kw = dict(num_workers=2, pin_memory=(cfg.DEVICE == 'cuda'))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)

    split_counts = compute_split_counts(train_ds, val_ds, test_ds, label_map)
    return train_loader, val_loader, test_loader, split_counts


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print(" DUAL TRANSFORMER – VSL TRAINING ".center(60, "="))
    print("="*60)
    print(f"\n  Device : {cfg.DEVICE}")
    print(f"  Input  : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")

    label_map   = load_label_map(cfg.LABEL_MAP_PATH)
    num_classes = len(label_map)
    print(f"  Classes: {num_classes} → {list(label_map.keys())}")

    train_loader, val_loader, test_loader, split_counts = \
        build_dataloaders(cfg.DATA_DIR, label_map, cfg)

    model = DualTransformer(
        feat_dim=cfg.FEAT_DIM, seq_len=cfg.SEQ_LEN,
        num_classes=num_classes, cfg=cfg,
    )

    trainer = Trainer(model, train_loader, val_loader, test_loader,
                      label_map, cfg, split_counts=split_counts)
    trainer.train()
    trainer.evaluate_and_plot()

    print(f"\n  HOAN THANH!")
    print(f"  Bieu do : {cfg.CHART_DIR}/")
    print(f"  Index   : {cfg.CHART_DIR}/chart_index.json\n")


if __name__ == '__main__':
    main()