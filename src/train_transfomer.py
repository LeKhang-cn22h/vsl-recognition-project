"""
train.py - Hand-Aware Transformer cho VSL (208 dim)
====================================================
BƯỚC 3 trong pipeline

Architecture:
  Input (B, 64, 208) → FeatureParser → tách thành:
    ├─ pose       (B, 64, 45)
    ├─ left_hand  (B, 64, 21, 3)   ← xyz landmarks
    ├─ right_hand (B, 64, 21, 3)
    ├─ curl       (B, 64, 30)      ← finger curl features
    └─ emotion    (B, 64, 7)

  ┌─ FingerEncoder        : encode từng ngón riêng biệt → (B,T,6,finger_dim)
  ├─ CurlEncoder          : encode curl features → (B,T,curl_embed_dim)
  ├─ FingerGraphAttention : GAT học quan hệ không gian giữa 6 nhóm ngón
  ├─ HandCrossAttention   : cross-attention học quan hệ tay trái ↔ tay phải
  ├─ HandAggregator       : gộp finger + curl + pose + emotion → temporal input
  └─ TemporalTransformer  : self-attention theo thời gian → logits

Biểu đồ tự động xuất sau training (logs/plots/):
  1_training_curves.png      — loss + accuracy (train vs val) theo epoch
  2_lr_schedule.png          — learning rate warmup + cosine decay
  3_confusion_matrix.png     — heatmap N×N trên test set
  4_per_class_metrics.png    — precision / recall / F1 từng ký hiệu VSL
  5_data_distribution.png    — số sample mỗi class (train/val/test)
  6_architecture_diagram.png — pipeline với tensor shapes
  7_attention_weights.png    — temporal pooling weights mẫu test
  8_training_summary.png     — dashboard tổng hợp 1 trang

Chạy:
  python train.py
  python train.py --epochs 120 --batch_size 32
  python train.py --finger_dim 64 --temporal_layers 6
  python train.py --data_dir data/processed
  python train.py --no_plots          # bỏ qua biểu đồ
"""

import os
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
from torch.utils.data import Dataset, DataLoader

# ── Matplotlib backend không cần GUI ──────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Style nhất quán toàn bộ biểu đồ
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

PLOT_DIR = None   # sẽ set trong main()


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    # ── Data ──────────────────────────────────────────────────────
    DATA_DIR   = "Khang_processed/processed"
    SEQ_LEN    = 64
    FEAT_DIM   = 208   # pose(45) + hands_xyz(126) + curl(30) + emotion(7)

    # ── Layout (nhất quán với video_to_npy.py) ────────────────────
    POSE_DIM   = 45
    HAND_DIM   = 63    # 21 lm × 3 (1 tay)
    CURL_DIM   = 30    # 5 ngón × 3 feat × 2 tay
    EMO_DIM    = 7

    # ── Cấu trúc ngón tay MediaPipe ───────────────────────────────
    # Wrist: 0
    # Thumb:  [1,2,3,4]   Index: [5,6,7,8]   Middle: [9,10,11,12]
    # Ring:   [13,14,15,16]  Pinky: [17,18,19,20]
    FINGER_GROUPS = {
        'wrist':  [0],
        'thumb':  [1, 2, 3, 4],
        'index':  [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring':   [13, 14, 15, 16],
        'pinky':  [17, 18, 19, 20],
    }
    FINGER_NAMES = list(FINGER_GROUPS.keys())   # 6 nhóm: wrist + 5 ngón
    NUM_FINGERS  = len(FINGER_GROUPS)           # 6

    # ── Graph edges (kết nối giữa các nhóm) ──────────────────────
    FINGER_EDGES = [
        (0,1),(0,2),(0,3),(0,4),(0,5),   # wrist → mỗi ngón
        (1,2),(2,3),(3,4),(4,5),          # ngón kề nhau
    ]

    CURL_LEFT_DIM  = 15
    CURL_RIGHT_DIM = 15

    # ── Model architecture ────────────────────────────────────────
    FINGER_DIM      = 32
    CURL_EMBED_DIM  = 32
    GRAPH_HEADS     = 4
    TEMPORAL_DIM    = 256
    TEMPORAL_HEADS  = 8
    TEMPORAL_LAYERS = 4
    DROPOUT         = 0.2

    # ── Training ──────────────────────────────────────────────────
    EPOCHS       = 120
    BATCH_SIZE   = 32
    LR           = 5e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 20
    GRAD_CLIP    = 1.0
    LR_WARMUP    = 8
    LR_MIN       = 1e-6

    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR        = "logs"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class VSLDataset(Dataset):
    def __init__(self, data_dir, label_map, split='train', augment=False):
        self.samples = []
        self.augment = augment

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  [WARN] Không tìm thấy: {split_dir}")
            return

        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fp in sorted(Path(label_dir).glob('*.npy')):
                self.samples.append((str(fp), label_idx))

        print(f"  [{split:5s}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)   # (64, 208)

        if self.augment and np.random.rand() < 0.4:
            data[:, 45:171]  += np.random.normal(0, 0.002, (64, 126)).astype(np.float32)
            data[:, 171:201] += np.random.normal(0, 0.005, (64, 30)).astype(np.float32)
            data[:, 171:201]  = np.clip(data[:, 171:201], 0.0, 1.0)

        return torch.from_numpy(data), label


def load_label_map(data_dir):
    path = os.path.join(data_dir, 'label_map.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy: {path}\nChạy video_to_npy.py trước!")
    with open(path) as f:
        return json.load(f)


def build_dataloaders(cfg):
    label_map = load_label_map(cfg.DATA_DIR)
    print(f"\n  Labels ({len(label_map)}): {list(label_map.keys())}")

    train_ds = VSLDataset(cfg.DATA_DIR, label_map, 'train', augment=True)
    val_ds   = VSLDataset(cfg.DATA_DIR, label_map, 'val',   augment=False)
    test_ds  = VSLDataset(cfg.DATA_DIR, label_map, 'test',  augment=False)

    if len(train_ds) == 0:
        raise ValueError("Train dataset trống!")

    kw = dict(num_workers=0, pin_memory=(cfg.DEVICE == 'cuda'))
    return (
        DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
        label_map,
        {'train': train_ds, 'val': val_ds, 'test': test_ds},
    )


# ═══════════════════════════════════════════════════════════════════
# FEATURE PARSER
# ═══════════════════════════════════════════════════════════════════

class FeatureParser:
    @staticmethod
    def parse(x: torch.Tensor):
        B, T, _ = x.shape
        pose        = x[:, :, 0:45]
        left_flat   = x[:, :, 45:108]
        right_flat  = x[:, :, 108:171]
        curl_left   = x[:, :, 171:186]
        curl_right  = x[:, :, 186:201]
        emotion     = x[:, :, 201:208]
        left_hand   = left_flat.view(B, T, 21, 3)
        right_hand  = right_flat.view(B, T, 21, 3)
        return pose, left_hand, right_hand, curl_left, curl_right, emotion


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: FINGER ENCODER
# ═══════════════════════════════════════════════════════════════════

class FingerEncoder(nn.Module):
    def __init__(self, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.finger_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']

        self.register_buffer('wrist_idx',  torch.tensor([0], dtype=torch.long))
        self.register_buffer('thumb_idx',  torch.tensor([1, 2, 3, 4], dtype=torch.long))
        self.register_buffer('index_idx',  torch.tensor([5, 6, 7, 8], dtype=torch.long))
        self.register_buffer('middle_idx', torch.tensor([9, 10, 11, 12], dtype=torch.long))
        self.register_buffer('ring_idx',   torch.tensor([13, 14, 15, 16], dtype=torch.long))
        self.register_buffer('pinky_idx',  torch.tensor([17, 18, 19, 20], dtype=torch.long))

        self.encoders = nn.ModuleDict()
        for fname in self.finger_names:
            idx_tensor = getattr(self, f"{fname}_idx")
            in_dim = len(idx_tensor) * 3
            self.encoders[fname] = nn.Sequential(
                nn.Linear(in_dim, out_dim * 2),
                nn.LayerNorm(out_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        B, T, _, _ = hand.shape
        vecs = []
        for fname in self.finger_names:
            idx  = getattr(self, f"{fname}_idx")
            flat = hand[:, :, idx, :].reshape(B, T, -1)
            vecs.append(self.encoders[fname](flat))
        return torch.stack(vecs, dim=2)


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: CURL ENCODER
# ═══════════════════════════════════════════════════════════════════

class CurlEncoder(nn.Module):
    def __init__(self, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.out_dim   = out_dim
        self.n_fingers = 5
        per_dim   = out_dim // self.n_fingers
        remainder = out_dim - per_dim * (self.n_fingers - 1)

        self.finger_encs = nn.ModuleList()
        for i in range(self.n_fingers):
            o = remainder if i == self.n_fingers - 1 else per_dim
            self.finger_encs.append(nn.Sequential(
                nn.Linear(3, 16), nn.GELU(), nn.Linear(16, o),
            ))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, curl: torch.Tensor) -> torch.Tensor:
        parts = []
        for i, enc in enumerate(self.finger_encs):
            parts.append(enc(curl[:, :, i*3 : i*3+3]))
        return self.out_proj(torch.cat(parts, dim=-1))


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: FINGER GRAPH ATTENTION
# ═══════════════════════════════════════════════════════════════════

class FingerGraphAttention(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim phải chia hết cho num_heads"
        self.feat_dim  = feat_dim
        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads

        adj = torch.zeros(cfg.NUM_FINGERS, cfg.NUM_FINGERS)
        for i, j in cfg.FINGER_EDGES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = adj + torch.eye(cfg.NUM_FINGERS)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        self.register_buffer('adj', adj / deg)

        self.W_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_o = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        BT = B * T
        xf = x.reshape(BT, N, D)
        Q = self.W_q(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        mask   = self.adj.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V).transpose(1, 2).reshape(BT, N, D)
        return self.norm(self.W_o(out).reshape(B, T, N, D) + x)


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: HAND CROSS ATTENTION
# ═══════════════════════════════════════════════════════════════════

class HandCrossAttention(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_lr = nn.MultiheadAttention(feat_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.cross_rl = nn.MultiheadAttention(feat_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.norm_l   = nn.LayerNorm(feat_dim)
        self.norm_r   = nn.LayerNorm(feat_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        B, T, N, D = left.shape
        BT = B * T
        lf = left.reshape(BT, N, D)
        rf = right.reshape(BT, N, D)
        l_ctx, _ = self.cross_lr(lf, rf, rf)
        r_ctx, _ = self.cross_rl(rf, lf, lf)
        return (self.norm_l(lf + l_ctx).reshape(B, T, N, D),
                self.norm_r(rf + r_ctx).reshape(B, T, N, D))


# ═══════════════════════════════════════════════════════════════════
# MODULE 5: HAND AGGREGATOR
# ═══════════════════════════════════════════════════════════════════

class HandAggregator(nn.Module):
    def __init__(self, finger_dim, curl_embed_dim, pose_dim, emo_dim,
                 out_dim, dropout=0.1):
        super().__init__()
        self.finger_pool = nn.Linear(cfg.NUM_FINGERS, 1, bias=False)
        total_in = finger_dim * 2 + curl_embed_dim * 2 + pose_dim + emo_dim
        self.proj = nn.Sequential(
            nn.Linear(total_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, left_fingers, right_fingers,
                curl_left_emb, curl_right_emb, pose, emotion):
        l_pool = self.finger_pool(left_fingers.transpose(-1, -2)).squeeze(-1)
        r_pool = self.finger_pool(right_fingers.transpose(-1, -2)).squeeze(-1)
        combined = torch.cat(
            [l_pool, r_pool, curl_left_emb, curl_right_emb, pose, emotion], dim=-1)
        return self.proj(combined)


# ═══════════════════════════════════════════════════════════════════
# MODULE 6: TEMPORAL TRANSFORMER
# ═══════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    def __init__(self, model_dim, num_heads=8, num_layers=4,
                 dropout=0.1, max_len=256):
        super().__init__()
        pe  = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, model_dim, 2).float()
                        * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.norm(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════
# MODEL CHÍNH: HandAwareVSLClassifier
# ═══════════════════════════════════════════════════════════════════

class HandAwareVSLClassifier(nn.Module):
    def __init__(self, num_classes: int, cfg=cfg):
        super().__init__()
        self.cfg = cfg or Config()
        self.finger_encoder = FingerEncoder(
            out_dim=self.cfg.FINGER_DIM, dropout=self.cfg.DROPOUT)
        self.curl_encoder = CurlEncoder(
            out_dim=cfg.CURL_EMBED_DIM, dropout=cfg.DROPOUT)
        self.graph_attn_left  = FingerGraphAttention(
            feat_dim=cfg.FINGER_DIM, num_heads=cfg.GRAPH_HEADS, dropout=cfg.DROPOUT)
        self.graph_attn_right = FingerGraphAttention(
            feat_dim=cfg.FINGER_DIM, num_heads=cfg.GRAPH_HEADS, dropout=cfg.DROPOUT)
        self.cross_attn = HandCrossAttention(
            feat_dim=cfg.FINGER_DIM, num_heads=cfg.GRAPH_HEADS, dropout=cfg.DROPOUT)
        self.aggregator = HandAggregator(
            finger_dim    = cfg.FINGER_DIM,
            curl_embed_dim= cfg.CURL_EMBED_DIM,
            pose_dim      = cfg.POSE_DIM,
            emo_dim       = cfg.EMO_DIM,
            out_dim       = cfg.TEMPORAL_DIM,
            dropout       = cfg.DROPOUT)
        self.temporal = TemporalTransformer(
            model_dim  = cfg.TEMPORAL_DIM,
            num_heads  = cfg.TEMPORAL_HEADS,
            num_layers = cfg.TEMPORAL_LAYERS,
            dropout    = cfg.DROPOUT)
        self.temporal_pool = nn.Linear(cfg.TEMPORAL_DIM, 1)
        mid = max(num_classes * 4, 256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.TEMPORAL_DIM),
            nn.Linear(cfg.TEMPORAL_DIM, mid),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT / 2),
            nn.Linear(mid // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pose, left_xyz, right_xyz, curl_left, curl_right, emotion = \
            FeatureParser.parse(x)
        left_fingers  = self.finger_encoder(left_xyz)
        right_fingers = self.finger_encoder(right_xyz)
        curl_left_emb  = self.curl_encoder(curl_left)
        curl_right_emb = self.curl_encoder(curl_right)
        left_fingers  = self.graph_attn_left(left_fingers)
        right_fingers = self.graph_attn_right(right_fingers)
        left_fingers, right_fingers = self.cross_attn(left_fingers, right_fingers)
        temporal_in  = self.aggregator(
            left_fingers, right_fingers,
            curl_left_emb, curl_right_emb, pose, emotion)
        temporal_out = self.temporal(temporal_in)
        weights  = torch.softmax(
            self.temporal_pool(temporal_out).squeeze(-1), dim=-1)
        context  = (temporal_out * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(context)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về temporal attention weights (B, T) để visualize."""
        pose, left_xyz, right_xyz, curl_left, curl_right, emotion = \
            FeatureParser.parse(x)
        left_fingers  = self.finger_encoder(left_xyz)
        right_fingers = self.finger_encoder(right_xyz)
        curl_left_emb  = self.curl_encoder(curl_left)
        curl_right_emb = self.curl_encoder(curl_right)
        left_fingers  = self.graph_attn_left(left_fingers)
        right_fingers = self.graph_attn_right(right_fingers)
        left_fingers, right_fingers = self.cross_attn(left_fingers, right_fingers)
        temporal_in  = self.aggregator(
            left_fingers, right_fingers,
            curl_left_emb, curl_right_emb, pose, emotion)
        temporal_out = self.temporal(temporal_in)
        return torch.softmax(
            self.temporal_pool(temporal_out).squeeze(-1), dim=-1)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION MODULE
# 8 biểu đồ nghiên cứu — tự động xuất sau training
# ═══════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Xuất 8 biểu đồ khoa học vào logs/plots/.
    Gọi: viz = Visualizer(trainer); viz.plot_all(...)
    """

    C_TRAIN = '#2563EB'   # xanh dương — train
    C_VAL   = '#DC2626'   # đỏ         — val
    C_TEST  = '#16A34A'   # xanh lá    — test
    C_BEST  = '#D97706'   # cam        — best epoch

    def __init__(self, trainer):
        self.trainer    = trainer
        self.cfg        = trainer.cfg
        self.history    = trainer.history
        self.label_map  = trainer.label_map
        self.idx2label  = trainer.idx2label
        self.best_epoch = trainer.best_epoch
        self.plot_dir   = PLOT_DIR
        os.makedirs(self.plot_dir, exist_ok=True)
        print(f"\n  [VIZ] Xuất biểu đồ vào: {self.plot_dir}")

    # ── helpers ───────────────────────────────────────────────────

    def _save(self, fig, name, idx):
        path = os.path.join(self.plot_dir, f'{idx}_{name}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"  [{idx}] {name:<26} → {path}")
        return path

    # ─────────────────────────────────────────────────────────────
    # 1. Training curves
    # ─────────────────────────────────────────────────────────────
    def plot_training_curves(self):
        epochs  = range(1, len(self.history['train_loss']) + 1)
        n_ep    = len(epochs)
        best_ep = self.best_epoch

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Training Curves — HandAware VSL Classifier',
                     fontsize=14, fontweight='bold', y=1.01)

        # Loss
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'],
                color=self.C_TRAIN, lw=2, label='Train loss')
        ax.plot(epochs, self.history['val_loss'],
                color=self.C_VAL,   lw=2, label='Val loss')
        ax.axvline(best_ep, color=self.C_BEST, ls='--', lw=1.5,
                   label=f'Best epoch ({best_ep})')
        ax.scatter([best_ep], [self.history['val_loss'][best_ep - 1]],
                   color=self.C_BEST, zorder=5, s=70)
        ax.set_title('Cross-Entropy Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(framealpha=0.9)
        ax.set_xlim(1, n_ep)

        # Accuracy
        ax = axes[1]
        tr_pct = [a * 100 for a in self.history['train_acc']]
        vl_pct = [a * 100 for a in self.history['val_acc']]
        ax.plot(epochs, tr_pct, color=self.C_TRAIN, lw=2, label='Train acc')
        ax.plot(epochs, vl_pct, color=self.C_VAL,   lw=2, label='Val acc')
        ax.axvline(best_ep, color=self.C_BEST, ls='--', lw=1.5,
                   label=f'Best epoch ({best_ep})')
        bv = vl_pct[best_ep - 1]
        ax.scatter([best_ep], [bv], color=self.C_BEST, zorder=5, s=70)
        ax.annotate(f'{bv:.1f}%',
                    xy=(best_ep, bv),
                    xytext=(best_ep + max(1, n_ep * 0.03), bv),
                    fontsize=9, color=self.C_BEST)
        ax.set_title('Classification Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 108)
        ax.legend(framealpha=0.9)
        ax.set_xlim(1, n_ep)

        fig.tight_layout()
        return self._save(fig, 'training_curves', 1)

    # ─────────────────────────────────────────────────────────────
    # 2. Learning rate schedule
    # ─────────────────────────────────────────────────────────────
    def plot_lr_schedule(self):
        lrs    = self.history['lr']
        epochs = range(1, len(lrs) + 1)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, lrs, color='#7C3AED', lw=2)
        ax.fill_between(epochs, lrs, alpha=0.12, color='#7C3AED')

        warmup = self.cfg.LR_WARMUP
        ax.axvline(warmup, color='#6B7280', ls=':', lw=1.3)
        ax.text(warmup + 0.5, max(lrs) * 0.90,
                f'Warmup end\n(epoch {warmup})', fontsize=9, color='#6B7280')
        ax.axvline(self.best_epoch, color=self.C_BEST, ls='--', lw=1.5,
                   label=f'Best epoch ({self.best_epoch})')

        ax.set_title('Learning Rate Schedule (Linear Warmup + Cosine Decay)',
                     fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.legend(framealpha=0.9)
        ax.set_xlim(1, len(lrs))

        fig.tight_layout()
        return self._save(fig, 'lr_schedule', 2)

    # ─────────────────────────────────────────────────────────────
    # 3. Confusion matrix
    # ─────────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, all_labels, all_preds):
        labels_list = [self.idx2label[i] for i in range(len(self.label_map))]
        cm      = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        n       = len(labels_list)

        cell    = max(0.55, min(1.1, 9.0 / n))
        fig, ax = plt.subplots(figsize=(n * cell + 2.5, n * cell + 1.5))

        cmap = LinearSegmentedColormap.from_list(
            'vsl_cm', ['#F0F9FF', '#1D4ED8'])
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        thresh = 0.5
        fs_cell = max(6, min(10, int(90 / n)))
        for i in range(n):
            for j in range(n):
                val   = cm_norm[i, j]
                count = cm[i, j]
                clr   = 'white' if val > thresh else '#1F2937'
                ax.text(j, i, f'{val:.0%}\n({count})',
                        ha='center', va='center',
                        color=clr, fontsize=fs_cell, fontweight='bold')

        fs_tick = max(7, min(11, int(100 / n)))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=fs_tick)
        ax.set_yticklabels(labels_list, fontsize=fs_tick)
        ax.set_xlabel('Predicted label', fontsize=11)
        ax.set_ylabel('True label',      fontsize=11)
        ax.set_title('Confusion Matrix — Test Set (row-normalized)',
                     fontweight='bold', fontsize=13)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label('Recall rate', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig.tight_layout()
        return self._save(fig, 'confusion_matrix', 3)

    # ─────────────────────────────────────────────────────────────
    # 4. Per-class metrics
    # ─────────────────────────────────────────────────────────────
    def plot_per_class_metrics(self, all_labels, all_preds):
        labels_list = [self.idx2label[i] for i in range(len(self.label_map))]
        report = classification_report(
            all_labels, all_preds, target_names=labels_list,
            output_dict=True, zero_division=0)

        prec = [report[c]['precision'] * 100 for c in labels_list]
        rec  = [report[c]['recall']    * 100 for c in labels_list]
        f1   = [report[c]['f1-score']  * 100 for c in labels_list]
        n    = len(labels_list)
        x    = np.arange(n)
        w    = 0.26

        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5))
        b1 = ax.bar(x - w, prec, w, label='Precision', color='#3B82F6', alpha=0.9)
        b2 = ax.bar(x,     rec,  w, label='Recall',    color='#EF4444', alpha=0.9)
        b3 = ax.bar(x + w, f1,   w, label='F1-score',  color='#10B981', alpha=0.9)

        if n <= 20:
            for bar_group in [b1, b2, b3]:
                for bar in bar_group:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f'{h:.0f}', ha='center', va='bottom', fontsize=7)

        macro_f1 = report['macro avg']['f1-score'] * 100
        ax.axhline(macro_f1, color='#6B7280', ls='--', lw=1.2, alpha=0.7)
        ax.text(n - 0.5, macro_f1 + 1,
                f'Macro F1: {macro_f1:.1f}%', fontsize=9, color='#6B7280')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=40, ha='right',
                           fontsize=max(7, min(10, int(110 / n))))
        ax.set_ylabel('Score (%)')
        ax.set_ylim(0, 115)
        ax.set_title('Per-Class Metrics — Precision / Recall / F1-score (Test Set)',
                     fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, 'per_class_metrics', 4)

    # ─────────────────────────────────────────────────────────────
    # 5. Data distribution
    # ─────────────────────────────────────────────────────────────
    def plot_data_distribution(self, datasets):
        labels_list = [self.idx2label[i] for i in range(len(self.label_map))]

        def count(ds):
            c = {l: 0 for l in labels_list}
            for _, li in ds.samples:
                c[self.idx2label[li]] += 1
            return [c[l] for l in labels_list]

        tr = count(datasets['train'])
        vl = count(datasets['val'])
        te = count(datasets['test'])
        n  = len(labels_list)
        x  = np.arange(n)
        w  = 0.26

        fig, ax = plt.subplots(figsize=(max(10, n * 0.85), 5))
        ax.bar(x - w, tr, w, label='Train', color=self.C_TRAIN, alpha=0.85)
        ax.bar(x,     vl, w, label='Val',   color=self.C_VAL,   alpha=0.85)
        ax.bar(x + w, te, w, label='Test',  color=self.C_TEST,  alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=40, ha='right',
                           fontsize=max(7, min(10, int(110 / n))))
        ax.set_ylabel('Số lượng mẫu')
        ax.set_title('Data Distribution — Samples per Class (Train / Val / Test)',
                     fontweight='bold')
        ax.legend(framealpha=0.9)

        total = sum(tr) + sum(vl) + sum(te)
        ax.text(0.99, 0.97,
                f'Total: {total:,}  |  Train: {sum(tr):,}  |  '
                f'Val: {sum(vl):,}  |  Test: {sum(te):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color='#374151',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D1D5DB'))

        fig.tight_layout()
        return self._save(fig, 'data_distribution', 5)

    # ─────────────────────────────────────────────────────────────
    # 6. Architecture diagram
    # ─────────────────────────────────────────────────────────────
    def plot_architecture_diagram(self):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 6)
        ax.axis('off')
        fig.patch.set_facecolor('white')

        # (label, sublabel, xc, yc, w, h, fc, ec)
        blocks = [
            ('Input',
             f'(B,{cfg.SEQ_LEN},{cfg.FEAT_DIM})',
             1.1, 5.0, 1.6, 0.65, '#DBEAFE', '#2563EB'),
            ('Feature\nParser', '',
             3.0, 5.0, 1.5, 0.65, '#F3E8FF', '#7C3AED'),
            ('Pose & Emotion',
             f'({cfg.POSE_DIM}+{cfg.EMO_DIM})',
             1.1, 3.5, 1.5, 0.70, '#FEF9C3', '#CA8A04'),
            ('Left Hand\nXYZ', '(21×3)',
             3.0, 3.5, 1.4, 0.70, '#DCFCE7', '#16A34A'),
            ('Right Hand\nXYZ', '(21×3)',
             4.8, 3.5, 1.4, 0.70, '#DCFCE7', '#16A34A'),
            ('Curl L/R',
             f'(2×{cfg.CURL_DIM//2})',
             6.6, 3.5, 1.3, 0.70, '#FFE4E6', '#DC2626'),
            ('FingerEncoder\n(×2)',
             f'→(6,{cfg.FINGER_DIM})',
             3.9, 2.2, 1.8, 0.70, '#DCFCE7', '#15803D'),
            ('CurlEncoder\n(×2)',
             f'→({cfg.CURL_EMBED_DIM})',
             6.6, 2.2, 1.5, 0.70, '#FFE4E6', '#B91C1C'),
            ('FingerGraph\nAttention',
             f'GAT {cfg.GRAPH_HEADS}H',
             3.0, 1.1, 1.7, 0.75, '#F0FFF4', '#166534'),
            ('Hand Cross\nAttention',
             'L↔R bidirectional',
             5.0, 1.1, 1.8, 0.75, '#F0FFF4', '#166534'),
            ('Hand\nAggregator',
             f'→({cfg.TEMPORAL_DIM})',
             7.5, 2.2, 1.7, 0.70, '#FEF3C7', '#D97706'),
            ('Temporal\nTransformer',
             f'{cfg.TEMPORAL_LAYERS}L×{cfg.TEMPORAL_HEADS}H pre-LN',
             9.6, 2.2, 1.9, 0.80, '#EDE9FE', '#6D28D9'),
            ('Attention\nPooling',
             '(B,T,D)→(B,D)',
             11.7, 2.2, 1.5, 0.70, '#E0E7FF', '#4338CA'),
            ('Classifier\nMLP',
             '→num_classes',
             13.5, 2.2, 1.6, 0.70, '#FEE2E2', '#991B1B'),
        ]

        def draw_block(lbl, sub, xc, yc, w, h, fc, ec):
            rect = mpatches.FancyBboxPatch(
                (xc - w/2, yc - h/2), w, h,
                boxstyle='round,pad=0.06',
                facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2)
            ax.add_patch(rect)
            main_y = yc + (0.10 if sub else 0)
            ax.text(xc, main_y, lbl, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#1F2937',
                    multialignment='center', zorder=3)
            if sub:
                ax.text(xc, yc - 0.17, sub, ha='center', va='center',
                        fontsize=6.5, color='#6B7280',
                        multialignment='center', zorder=3)

        for b in blocks:
            draw_block(*b)

        def arr(x0, y0, x1, y1):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#9CA3AF',
                                        lw=1.2),
                        zorder=1)

        # Input → Parser
        arr(1.9, 5.0, 2.25, 5.0)
        # Parser → branches
        arr(3.0, 4.67, 1.1, 3.85)
        arr(3.0, 4.67, 3.0, 3.85)
        arr(3.0, 4.67, 4.8, 3.85)
        arr(3.0, 4.67, 6.6, 3.85)
        # Hands → FingerEncoder
        arr(3.0, 3.15, 3.9, 2.55)
        arr(4.8, 3.15, 3.9, 2.55)
        # Curl → CurlEncoder
        arr(6.6, 3.15, 6.6, 2.55)
        # FingerEncoder → GraphAttn
        arr(3.9, 1.85, 3.0, 1.48)
        # GraphAttn → CrossAttn
        arr(3.85, 1.10, 4.10, 1.10)
        # CrossAttn → Aggregator
        arr(5.0, 1.48, 7.5, 1.85)
        # CurlEncoder → Aggregator
        arr(6.6, 1.85, 7.35, 2.20)
        # Pose → Aggregator
        arr(1.1, 3.15, 7.30, 2.48)
        # Aggregator → Temporal
        arr(8.35, 2.20, 8.65, 2.20)
        # Temporal → Pool
        arr(10.55, 2.20, 10.95, 2.20)
        # Pool → Clf
        arr(12.45, 2.20, 12.70, 2.20)

        ax.set_title(
            f'HandAwareVSLClassifier — Architecture Pipeline  '
            f'[finger_dim={cfg.FINGER_DIM} | temporal_dim={cfg.TEMPORAL_DIM} | '
            f'{cfg.TEMPORAL_LAYERS}L×{cfg.TEMPORAL_HEADS}H]',
            fontsize=11, fontweight='bold', pad=10)

        fig.tight_layout()
        return self._save(fig, 'architecture_diagram', 6)

    # ─────────────────────────────────────────────────────────────
    # 7. Temporal attention weight heatmap
    # ─────────────────────────────────────────────────────────────
    def plot_attention_weights(self, model, test_loader):
        model.eval()
        collected_w, collected_y = [], []
        with torch.no_grad():
            for x, y in test_loader:
                w = model.get_attention_weights(x.to(self.cfg.DEVICE))
                collected_w.append(w.cpu())
                collected_y.extend(y.tolist())
                if len(collected_y) >= 16:
                    break

        n_show  = min(16, len(collected_y))
        weights = torch.cat(collected_w, dim=0)[:n_show].numpy()
        labels  = [self.idx2label[i] for i in collected_y[:n_show]]

        fig, ax = plt.subplots(figsize=(13, max(4, n_show * 0.35)))
        cmap = LinearSegmentedColormap.from_list(
            'attn', ['#F9FAFB', '#2563EB', '#1E1B4B'])
        im = ax.imshow(weights, aspect='auto', cmap=cmap,
                       vmin=0, vmax=weights.max())

        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f'Frame index (0 → {cfg.SEQ_LEN - 1})', fontsize=10)
        ax.set_ylabel('Test sample (true label)', fontsize=10)
        ax.set_title(
            'Temporal Attention Weights — Frame Importance (Darker = More Attended)',
            fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Attention weight', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        for i in range(1, n_show):
            ax.axhline(i - 0.5, color='white', lw=0.5)

        fig.tight_layout()
        return self._save(fig, 'attention_weights', 7)

    # ─────────────────────────────────────────────────────────────
    # 8. Training summary dashboard (1 trang tổng hợp)
    # ─────────────────────────────────────────────────────────────
    def plot_training_summary(self, all_labels, all_preds):
        epochs  = range(1, len(self.history['train_loss']) + 1)
        n_ep    = len(epochs)
        best_ep = self.best_epoch

        fig = plt.figure(figsize=(17, 10))
        fig.suptitle(
            'BarberGo VSL Recognition — HandAwareVSLClassifier Training Summary',
            fontsize=14, fontweight='bold', y=1.005)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

        # A: Loss ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(epochs, self.history['train_loss'],
                color=self.C_TRAIN, lw=1.8, label='Train')
        ax.plot(epochs, self.history['val_loss'],
                color=self.C_VAL,   lw=1.8, label='Val')
        ax.axvline(best_ep, color=self.C_BEST, ls='--', lw=1.3)
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.set_xlim(1, n_ep)

        # B: Accuracy ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(epochs, [a*100 for a in self.history['train_acc']],
                color=self.C_TRAIN, lw=1.8, label='Train')
        ax.plot(epochs, [a*100 for a in self.history['val_acc']],
                color=self.C_VAL,   lw=1.8, label='Val')
        ax.axvline(best_ep, color=self.C_BEST, ls='--', lw=1.3)
        bv = self.history['val_acc'][best_ep - 1] * 100
        ax.scatter([best_ep], [bv], color=self.C_BEST, s=55, zorder=5)
        ax.set_title(f'Accuracy  (best val = {bv:.1f}%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('%')
        ax.set_ylim(0, 108)
        ax.legend(fontsize=8)
        ax.set_xlim(1, n_ep)

        # C: LR ───────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(epochs, self.history['lr'], color='#7C3AED', lw=1.8)
        ax.fill_between(epochs, self.history['lr'],
                        alpha=0.12, color='#7C3AED')
        ax.axvline(best_ep, color=self.C_BEST, ls='--', lw=1.3)
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlim(1, n_ep)

        # D: Confusion matrix (mini) ───────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        labels_list = [self.idx2label[i] for i in range(len(self.label_map))]
        cm      = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        cmap2   = LinearSegmentedColormap.from_list('cm2', ['#F0F9FF', '#1D4ED8'])
        ax.imshow(cm_norm, cmap=cmap2, vmin=0, vmax=1, aspect='auto')
        n = len(labels_list)
        fs = max(5, min(8, int(60 / n)))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=fs)
        ax.set_yticklabels(labels_list, fontsize=fs)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)

        # E: Per-class F1 bar ─────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        report = classification_report(
            all_labels, all_preds, target_names=labels_list,
            output_dict=True, zero_division=0)
        f1_scores = [report[c]['f1-score'] * 100 for c in labels_list]
        colors    = ['#10B981' if s >= 80 else
                     '#F59E0B' if s >= 50 else '#EF4444'
                     for s in f1_scores]
        ax.barh(labels_list, f1_scores, color=colors, alpha=0.88)
        ax.axvline(80, color='#10B981', ls='--', lw=1.1, alpha=0.6)
        ax.axvline(50, color='#F59E0B', ls='--', lw=1.1, alpha=0.6)
        ax.set_xlim(0, 112)
        ax.set_xlabel('F1-score (%)')
        macro = report['macro avg']['f1-score'] * 100
        ax.set_title(f'Per-Class F1  (macro avg = {macro:.1f}%)')
        ax.tick_params(axis='y', labelsize=max(6, min(9, int(85 / n))))

        # F: Stats table ──────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        total_params = sum(p.numel() for p in self.trainer.model.parameters())
        test_acc = accuracy_score(all_labels, all_preds) * 100
        test_f1  = f1_score(all_labels, all_preds, average='macro',
                            zero_division=0) * 100
        test_pre = precision_score(all_labels, all_preds, average='macro',
                                   zero_division=0) * 100
        test_rec = recall_score(all_labels, all_preds, average='macro',
                                zero_division=0) * 100

        rows = [
            ['Metric',             'Value'],
            ['Best epoch',         f'{best_ep} / {self.cfg.EPOCHS}'],
            ['Best val accuracy',  f'{self.history["val_acc"][best_ep-1]*100:.2f}%'],
            ['Test accuracy',      f'{test_acc:.2f}%'],
            ['Macro precision',    f'{test_pre:.2f}%'],
            ['Macro recall',       f'{test_rec:.2f}%'],
            ['Macro F1',           f'{test_f1:.2f}%'],
            ['Num classes',        str(len(self.label_map))],
            ['Model parameters',   f'{total_params:,}'],
            ['Temporal dim',       str(self.cfg.TEMPORAL_DIM)],
            ['Transformer layers', f'{self.cfg.TEMPORAL_LAYERS}L × {self.cfg.TEMPORAL_HEADS}H'],
            ['Finger dim',         str(self.cfg.FINGER_DIM)],
            ['Curl embed dim',     str(self.cfg.CURL_EMBED_DIM)],
            ['Sequence length',    str(self.cfg.SEQ_LEN)],
            ['Batch size',         str(self.cfg.BATCH_SIZE)],
            ['LR (initial)',       f'{self.cfg.LR:.0e}'],
            ['Dropout',            str(self.cfg.DROPOUT)],
        ]

        tbl = ax.table(
            cellText=[[r[1]] for r in rows[1:]],
            rowLabels=[r[0] for r in rows[1:]],
            colLabels=['Value'],
            cellLoc='center', loc='center',
            bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#E5E7EB')
            if row == 0:
                cell.set_facecolor('#DBEAFE')
                cell.set_text_props(fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#F9FAFB')
            else:
                cell.set_facecolor('white')
        ax.set_title('Training Summary', fontweight='bold', pad=8)

        return self._save(fig, 'training_summary', 8)

    # ─────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────
    def plot_all(self, all_labels, all_preds, datasets, model, test_loader):
        print(f"\n{'='*55}")
        print(" XUẤT BIỂU ĐỒ NGHIÊN CỨU ".center(55, '='))
        print('='*55)
        self.plot_training_curves()
        self.plot_lr_schedule()
        self.plot_confusion_matrix(all_labels, all_preds)
        self.plot_per_class_metrics(all_labels, all_preds)
        self.plot_data_distribution(datasets)
        self.plot_architecture_diagram()
        self.plot_attention_weights(model, test_loader)
        self.plot_training_summary(all_labels, all_preds)
        print(f"\n  Tất cả biểu đồ đã lưu vào: {self.plot_dir}/")
        print('='*55)


# ═══════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 label_map, cfg, ckpt_name='best.pt'):
        self.model        = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.label_map    = label_map
        self.idx2label    = {v: k for k, v in label_map.items()}
        self.cfg          = cfg
        self.ckpt_name    = ckpt_name

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = self._build_scheduler()
        self.scaler    = (torch.cuda.amp.GradScaler()
                          if cfg.DEVICE == 'cuda' else None)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [], 'lr': []
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 1       # epoch có val_acc tốt nhất — dùng cho viz
        self.patience_cnt = 0

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)

    def _build_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.cfg.LR_WARMUP:
                return (epoch + 1) / self.cfg.LR_WARMUP
            progress = ((epoch - self.cfg.LR_WARMUP)
                        / max(1, self.cfg.EPOCHS - self.cfg.LR_WARMUP))
            return (self.cfg.LR_MIN / self.cfg.LR
                    + (1 - self.cfg.LR_MIN / self.cfg.LR)
                    * 0.5 * (1 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.cfg.DEVICE), y.to(self.cfg.DEVICE)
                if self.scaler and train:
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss   = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss   = self.criterion(logits, y)
                    if train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct    += (logits.argmax(-1) == y).sum().item()
                total      += y.size(0)
        return total_loss / total, correct / total

    def train(self):
        print(f"\n  Training on {self.cfg.DEVICE}...")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model params: {n_params:,}")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader,   train=False)
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)
            self.history['lr'].append(lr)

            print(f"  Ep {epoch:03d}/{self.cfg.EPOCHS} | "
                  f"Tr={tr_loss:.4f}/{tr_acc*100:.1f}% | "
                  f"Vl={vl_loss:.4f}/{vl_acc*100:.1f}% | "
                  f"LR={lr:.2e} | {time.time()-t0:.1f}s")

            if vl_acc > self.best_val_acc:
                self.best_val_acc = vl_acc
                self.best_epoch   = epoch
                self.patience_cnt = 0
                ckpt_path = os.path.join(self.cfg.CHECKPOINT_DIR, self.ckpt_name)
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'val_acc':     vl_acc,
                    'label_map':   self.label_map,
                    'model_type':  'HandAwareVSLClassifier',
                    'cfg': {
                        'FEAT_DIM':        self.cfg.FEAT_DIM,
                        'SEQ_LEN':         self.cfg.SEQ_LEN,
                        'FINGER_DIM':      self.cfg.FINGER_DIM,
                        'CURL_EMBED_DIM':  self.cfg.CURL_EMBED_DIM,
                        'TEMPORAL_DIM':    self.cfg.TEMPORAL_DIM,
                        'TEMPORAL_HEADS':  self.cfg.TEMPORAL_HEADS,
                        'TEMPORAL_LAYERS': self.cfg.TEMPORAL_LAYERS,
                        'GRAPH_HEADS':     self.cfg.GRAPH_HEADS,
                        'DROPOUT':         self.cfg.DROPOUT,
                    },
                }, ckpt_path)
                print(f"  → Saved best: {vl_acc*100:.2f}%")
            else:
                self.patience_cnt += 1

            if self.patience_cnt >= self.cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        log_path = os.path.join(self.cfg.LOG_DIR, 'history.json')
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"  History: {log_path}")

    def evaluate(self):
        """Evaluate trên test set, trả về (all_labels, all_preds) cho viz."""
        print(f"\n  Evaluating on test set...")

        # Load best checkpoint
        ckpt_path = os.path.join(self.cfg.CHECKPOINT_DIR, self.ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.cfg.DEVICE)
            self.model.load_state_dict(ckpt['model_state'])
            print(f"  Loaded best checkpoint "
                  f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']*100:.2f}%)")

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                preds = self.model(x.to(self.cfg.DEVICE)).argmax(-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())

        names  = [self.idx2label[i] for i in range(len(self.label_map))]
        report = classification_report(all_labels, all_preds,
                                       target_names=names, digits=4)
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)

        with open(os.path.join(self.cfg.LOG_DIR, 'report.txt'), 'w') as f:
            f.write(report)
        print(f"  Best val acc: {self.best_val_acc*100:.2f}%")
        return all_labels, all_preds


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global PLOT_DIR

    parser = argparse.ArgumentParser(
        description="Train HandAware VSL Classifier (208 dim — finger curl)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python train.py
  python train.py --epochs 100 --batch_size 16
  python train.py --finger_dim 64 --temporal_layers 6
  python train.py --data_dir data/processed
  python train.py --no_plots          # bỏ qua bước vẽ biểu đồ
        """
    )
    parser.add_argument("--data_dir",        default="data/processed")
    parser.add_argument("--epochs",          type=int,   default=120)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=5e-4)
    parser.add_argument("--finger_dim",      type=int,   default=32)
    parser.add_argument("--curl_embed_dim",  type=int,   default=32)
    parser.add_argument("--temporal_dim",    type=int,   default=256)
    parser.add_argument("--temporal_layers", type=int,   default=4)
    parser.add_argument("--temporal_heads",  type=int,   default=8)
    parser.add_argument("--graph_heads",     type=int,   default=4)
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--ckpt_name",       type=str,   default="best.pt")
    parser.add_argument("--no_plots",        action='store_true',
                        help="Bỏ qua bước vẽ biểu đồ (tiết kiệm thời gian)")
    args = parser.parse_args()

    # Cập nhật config
    cfg.DATA_DIR         = args.data_dir
    cfg.EPOCHS           = args.epochs
    cfg.BATCH_SIZE       = args.batch_size
    cfg.LR               = args.lr
    cfg.FINGER_DIM       = args.finger_dim
    cfg.CURL_EMBED_DIM   = args.curl_embed_dim
    cfg.TEMPORAL_DIM     = args.temporal_dim
    cfg.TEMPORAL_LAYERS  = args.temporal_layers
    cfg.TEMPORAL_HEADS   = args.temporal_heads
    cfg.GRAPH_HEADS      = args.graph_heads
    cfg.DROPOUT          = args.dropout

    # Set thư mục plot
    PLOT_DIR = os.path.join(cfg.LOG_DIR, 'plots')

    # Validate heads
    if cfg.FINGER_DIM % cfg.GRAPH_HEADS != 0:
        cfg.GRAPH_HEADS = 4
        print(f"  [WARN] Điều chỉnh graph_heads={cfg.GRAPH_HEADS}")
    if cfg.TEMPORAL_DIM % cfg.TEMPORAL_HEADS != 0:
        cfg.TEMPORAL_HEADS = 8
        print(f"  [WARN] Điều chỉnh temporal_heads={cfg.TEMPORAL_HEADS}")

    print("\n" + "=" * 60)
    print(" TRAIN HandAware VSL (208 dim) ".center(60, "="))
    print("=" * 60)
    print(f"\n  Device          : {cfg.DEVICE}")
    print(f"  Features        : {cfg.FEAT_DIM} dim")
    print(f"    pose={cfg.POSE_DIM}, hands_xyz={cfg.HAND_DIM*2}, "
          f"curl={cfg.CURL_DIM}, emotion={cfg.EMO_DIM}")
    print(f"\n  Finger groups   : {cfg.NUM_FINGERS} (wrist + 5 ngón)")
    print(f"  Finger dim      : {cfg.FINGER_DIM} per nhóm")
    print(f"  Curl embed      : {cfg.CURL_EMBED_DIM}")
    print(f"  Graph edges     : {len(cfg.FINGER_EDGES)} kết nối giải phẫu")
    print(f"  Temporal dim    : {cfg.TEMPORAL_DIM}")
    print(f"  Transformer     : {cfg.TEMPORAL_LAYERS}L × {cfg.TEMPORAL_HEADS}H")
    if not args.no_plots:
        print(f"\n  Biểu đồ đầu ra : {PLOT_DIR}/  (8 files)")
    print()

    # ── Build data ────────────────────────────────────────────────
    train_loader, val_loader, test_loader, label_map, datasets = \
        build_dataloaders(cfg)

    # ── Build model ───────────────────────────────────────────────
    model    = HandAwareVSLClassifier(num_classes=len(label_map), cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params    : {n_params:,}")
    print(f"\n  Architecture:")
    print(f"    FingerEncoder      : {cfg.NUM_FINGERS} nhóm × MLP ({cfg.FINGER_DIM} dim)")
    print(f"    CurlEncoder        : 5 ngón × 3 feat → {cfg.CURL_EMBED_DIM} dim (per tay)")
    print(f"    FingerGraphAttn    : GAT {cfg.GRAPH_HEADS} heads, adj mask giải phẫu")
    print(f"    HandCrossAttn      : bidirectional cross-attention 2 tay")
    print(f"    HandAggregator     : learned finger pool → {cfg.TEMPORAL_DIM} dim")
    print(f"    TemporalTransformer: {cfg.TEMPORAL_LAYERS}L × {cfg.TEMPORAL_HEADS}H, pre-norm")
    print(f"    TemporalAttnPool   : learned frame importance weights")

    # ── Train ─────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, test_loader,
                      label_map, cfg, ckpt_name=args.ckpt_name)
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────
    all_labels, all_preds = trainer.evaluate()

    # ── Visualize ─────────────────────────────────────────────────
    if not args.no_plots:
        viz = Visualizer(trainer)
        viz.plot_all(
            all_labels  = all_labels,
            all_preds   = all_preds,
            datasets    = datasets,
            model       = trainer.model,
            test_loader = test_loader,
        )

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"  Checkpoint : {cfg.CHECKPOINT_DIR}/{args.ckpt_name}")
    print(f"  Log        : {cfg.LOG_DIR}/history.json")
    print(f"  Report     : {cfg.LOG_DIR}/report.txt")
    if not args.no_plots:
        print(f"  Plots (8)  : {PLOT_DIR}/")
        print(f"    1_training_curves.png      — loss & accuracy curves")
        print(f"    2_lr_schedule.png          — warmup + cosine decay")
        print(f"    3_confusion_matrix.png     — N×N heatmap test set")
        print(f"    4_per_class_metrics.png    — precision/recall/F1 per class")
        print(f"    5_data_distribution.png    — sample count train/val/test")
        print(f"    6_architecture_diagram.png — pipeline tensor shapes")
        print(f"    7_attention_weights.png    — temporal frame importance")
        print(f"    8_training_summary.png     — dashboard 1 trang tổng hợp")
    print('='*60)


if __name__ == "__main__":
    main()