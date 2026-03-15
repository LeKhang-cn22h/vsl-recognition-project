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

Ưu điểm so với BiLSTM thông thường:
  - Hiểu cấu trúc bàn tay: từng ngón encode riêng
  - Biết quan hệ ngón-ngón: graph attention trên đồ thị ngón tay
  - Biết quan hệ 2 tay: cross-attention left↔right
  - Dùng finger curl: phân biệt "ngón co / ngón duỗi" rõ ràng
  - Temporal Transformer thay BiLSTM: song song hóa tốt hơn

Chạy:
  python train.py
  python train.py --epochs 120 --batch_size 32
  python train.py --finger_dim 64 --temporal_layers 6
  python train.py --data_dir data/processed
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
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    # ── Data ──────────────────────────────────────────────────────
    DATA_DIR   = "data/processed"
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
    # wrist (0) kết nối tất cả ngón; ngón liền nhau kết nối nhau
    FINGER_EDGES = [
        (0,1),(0,2),(0,3),(0,4),(0,5),   # wrist → mỗi ngón
        (1,2),(2,3),(3,4),(4,5),          # ngón kề nhau
    ]

    # ── Curl feature layout (input dim cho CurlEncoder) ───────────
    # Left hand:  curl[0:15]   Right hand: curl[15:30]
    CURL_LEFT_DIM  = 15   # 5 ngón × 3 feat
    CURL_RIGHT_DIM = 15

    # ── Model architecture ────────────────────────────────────────
    FINGER_DIM      = 32    # embed dim mỗi nhóm ngón sau FingerEncoder
    CURL_EMBED_DIM  = 32    # embed dim sau CurlEncoder
    GRAPH_HEADS     = 4     # heads trong FingerGraphAttention
    TEMPORAL_DIM    = 256   # model dim của TemporalTransformer
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
            # Augment nhẹ phần xyz (45:171) và curl (171:201)
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
    )


# ═══════════════════════════════════════════════════════════════════
# FEATURE PARSER
# Tách (B, T, 208) → các thành phần có cấu trúc
# ═══════════════════════════════════════════════════════════════════

class FeatureParser:
    """
    Tách tensor (B, T, 208) thành:
      pose       (B, T, 45)
      left_hand  (B, T, 21, 3)
      right_hand (B, T, 21, 3)
      curl_left  (B, T, 15)     ← 5 ngón × 3 feat, tay trái
      curl_right (B, T, 15)     ← 5 ngón × 3 feat, tay phải
      emotion    (B, T, 7)
    """

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
# Encode từng nhóm ngón tay riêng biệt → đặc trưng cục bộ từng ngón
# ═══════════════════════════════════════════════════════════════════

class FingerEncoder(nn.Module):
    """
    Encode 6 nhóm ngón (wrist + 5 ngón) - TorchScript compatible
    """
    def __init__(self, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        
        # CÁCH 1: Hardcode indices (đơn giản, dễ hiểu)
        # Định nghĩa finger groups ngay trong init
        self.finger_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
        
        # Register buffers để TorchScript biết đây là constants
        self.register_buffer('wrist_idx',  torch.tensor([0], dtype=torch.long))
        self.register_buffer('thumb_idx',  torch.tensor([1, 2, 3, 4], dtype=torch.long))
        self.register_buffer('index_idx',  torch.tensor([5, 6, 7, 8], dtype=torch.long))
        self.register_buffer('middle_idx', torch.tensor([9, 10, 11, 12], dtype=torch.long))
        self.register_buffer('ring_idx',   torch.tensor([13, 14, 15, 16], dtype=torch.long))
        self.register_buffer('pinky_idx',  torch.tensor([17, 18, 19, 20], dtype=torch.long))
        
        # Tạo encoders cho từng ngón
        self.encoders = nn.ModuleDict()
        for fname in self.finger_names:
            # Lấy indices tương ứng
            idx_tensor = getattr(self, f"{fname}_idx")
            in_dim = len(idx_tensor) * 3  # mỗi landmark có 3 tọa độ xyz
            
            self.encoders[fname] = nn.Sequential(
                nn.Linear(in_dim, out_dim * 2),
                nn.LayerNorm(out_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        """hand: (B, T, 21, 3) → (B, T, 6, out_dim)"""
        B, T, _, _ = hand.shape
        vecs = []
        
        # Dùng self.finger_names thay vì cfg
        for fname in self.finger_names:
            # Lấy indices từ buffer đã đăng ký
            idx = getattr(self, f"{fname}_idx")
            # Chọn landmarks và flatten
            flat = hand[:, :, idx, :].reshape(B, T, -1)
            # Encode
            vecs.append(self.encoders[fname](flat))
            
        return torch.stack(vecs, dim=2)


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: CURL ENCODER
# Encode finger curl features → biểu diễn trạng thái co/duỗi ngón
# ═══════════════════════════════════════════════════════════════════

class CurlEncoder(nn.Module):
    """
    Encode finger curl features - TorchScript compatible
    """
    def __init__(self, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.n_fingers = 5  # thumb, index, middle, ring, pinky
        
        # Tính toán dimensions
        per_dim = out_dim // self.n_fingers
        remainder = out_dim - per_dim * (self.n_fingers - 1)
        
        # Tạo encoders cho từng ngón
        self.finger_encs = nn.ModuleList()
        for i in range(self.n_fingers):
            o = remainder if i == self.n_fingers - 1 else per_dim
            self.finger_encs.append(nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, o),
            ))
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, curl: torch.Tensor) -> torch.Tensor:
        """curl: (B, T, 15) → (B, T, out_dim)"""
        B, T, _ = curl.shape
        parts = []
        
        for i, enc in enumerate(self.finger_encs):
            # Mỗi ngón có 3 features
            finger_feat = curl[:, :, i*3 : i*3+3]
            parts.append(enc(finger_feat))
            
        combined = torch.cat(parts, dim=-1)
        return self.out_proj(combined)


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: FINGER GRAPH ATTENTION
# Học quan hệ không gian giữa các nhóm ngón (graph message passing)
# ═══════════════════════════════════════════════════════════════════

class FingerGraphAttention(nn.Module):
    """
    Graph Attention Network (GAT) trên đồ thị 6 node (wrist + 5 ngón).
    Mỗi node = 1 nhóm ngón; cạnh = kết nối giải phẫu thực tế.

    Ý nghĩa: học "ngón trỏ đang co ảnh hưởng thế nào đến trạng thái
    ngón giữa / ngón cái" — quan hệ không gian giữa các ngón.

    Input:  (B, T, 6, feat_dim)
    Output: (B, T, 6, feat_dim)   — enriched bằng context các ngón khác
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim phải chia hết cho num_heads"
        self.feat_dim  = feat_dim
        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads

        # Adjacency matrix cố định (6×6), đối xứng
        adj = torch.zeros(cfg.NUM_FINGERS, cfg.NUM_FINGERS)
        for i, j in cfg.FINGER_EDGES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = adj + torch.eye(cfg.NUM_FINGERS)              # self-loop
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        self.register_buffer('adj', adj / deg)               # (6, 6) normalized

        self.W_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_o = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 6, feat_dim)"""
        B, T, N, D = x.shape
        BT = B * T
        xf = x.reshape(BT, N, D)

        Q = self.W_q(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (BT, heads, N, N)

        # Mask: chỉ attend qua cạnh đồ thị (kết nối giải phẫu)
        mask   = self.adj.unsqueeze(0).unsqueeze(0)              # (1, 1, 6, 6)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(BT, N, D)
        out = self.W_o(out).reshape(B, T, N, D)

        return self.norm(out + x)


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: HAND CROSS ATTENTION
# Học quan hệ giữa tay trái và tay phải
# ═══════════════════════════════════════════════════════════════════

class HandCrossAttention(nn.Module):
    """
    Cross-attention song phương giữa tay trái và tay phải.
    Giúp model học: "khi tay trái ở trạng thái X, tay phải thường ở trạng thái Y"
    — đặc biệt quan trọng cho ký hiệu 2 tay phối hợp.

    Input:  left  (B, T, 6, feat_dim)
            right (B, T, 6, feat_dim)
    Output: left_enriched, right_enriched — cùng shape
    """

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

        l_ctx, _ = self.cross_lr(lf, rf, rf)   # tay trái attend tay phải
        r_ctx, _ = self.cross_rl(rf, lf, lf)   # tay phải attend tay trái

        l_out = self.norm_l(lf + l_ctx).reshape(B, T, N, D)
        r_out = self.norm_r(rf + r_ctx).reshape(B, T, N, D)
        return l_out, r_out


# ═══════════════════════════════════════════════════════════════════
# MODULE 5: HAND AGGREGATOR
# Gộp tất cả spatial features → input cho Temporal Transformer
# ═══════════════════════════════════════════════════════════════════

class HandAggregator(nn.Module):
    """
    Tổng hợp:
      - finger xyz features (2 tay, 6 nhóm mỗi tay)
      - curl features (2 tay đã encode)
      - pose features
      - emotion features
    → (B, T, temporal_dim)

    Dùng learned pooling (Linear(6→1)) thay mean pool để có trọng số.
    """

    def __init__(self, finger_dim, curl_embed_dim, pose_dim, emo_dim,
                 out_dim, dropout=0.1):
        super().__init__()
        # Học trọng số pool qua 6 nhóm ngón
        self.finger_pool = nn.Linear(cfg.NUM_FINGERS, 1, bias=False)

        total_in = (finger_dim * 2      # left_pooled + right_pooled
                    + curl_embed_dim * 2 # curl_left_embed + curl_right_embed
                    + pose_dim           # 45
                    + emo_dim)           # 7

        self.proj = nn.Sequential(
            nn.Linear(total_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, left_fingers, right_fingers,
                curl_left_emb, curl_right_emb,
                pose, emotion):
        """
        left_fingers, right_fingers: (B, T, 6, finger_dim)
        curl_left_emb, curl_right_emb: (B, T, curl_embed_dim)
        pose:    (B, T, 45)
        emotion: (B, T, 7)
        → (B, T, out_dim)
        """
        # Weighted pool over 6 ngón: (B,T,6,D).T[-2] → Linear(6→1) → squeeze
        l_pool = self.finger_pool(
            left_fingers.transpose(-1, -2)).squeeze(-1)    # (B, T, finger_dim)
        r_pool = self.finger_pool(
            right_fingers.transpose(-1, -2)).squeeze(-1)

        combined = torch.cat(
            [l_pool, r_pool, curl_left_emb, curl_right_emb, pose, emotion],
            dim=-1)
        return self.proj(combined)


# ═══════════════════════════════════════════════════════════════════
# MODULE 6: TEMPORAL TRANSFORMER
# Self-attention theo chiều thời gian — học temporal dependencies
# ═══════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    Transformer encoder theo chiều thời gian với sinusoidal positional encoding.
    Dùng Pre-LN (norm_first=True) để training ổn định hơn Post-LN.

    Input:  (B, T, model_dim)
    Output: (B, T, model_dim)
    """

    def __init__(self, model_dim, num_heads=8, num_layers=4,
                 dropout=0.1, max_len=256):
        super().__init__()

        pe  = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, model_dim, 2).float()
                        * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, model_dim)

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
    """
    Pipeline đầy đủ cho VSL recognition với finger curl.

    (B,T,208)
      ↓ FeatureParser
    pose(B,T,45) | left_xyz(B,T,21,3) | right_xyz(B,T,21,3)
    curl_left(B,T,15) | curl_right(B,T,15) | emotion(B,T,7)
      ↓ FingerEncoder (6 nhóm ngón, encode riêng)
    left_fingers(B,T,6,finger_dim) | right_fingers(B,T,6,finger_dim)
      ↓ CurlEncoder (encode trạng thái co/duỗi)
    curl_left_emb(B,T,curl_embed_dim) | curl_right_emb(B,T,curl_embed_dim)
      ↓ FingerGraphAttention (quan hệ ngón-ngón trong 1 tay)
    left_enriched | right_enriched
      ↓ HandCrossAttention (quan hệ tay trái ↔ tay phải)
    left_cross | right_cross
      ↓ HandAggregator (gộp tất cả → temporal input)
    temporal_in(B,T,temporal_dim)
      ↓ TemporalTransformer (học temporal patterns)
    temporal_out(B,T,temporal_dim)
      ↓ Attention Pooling (frame nào quan trọng hơn)
    context(B,temporal_dim)
      ↓ Classifier
    logits(B,num_classes)
    """

    def __init__(self, num_classes: int, cfg=cfg):
        super().__init__()
        self.cfg = cfg or Config()
        self.finger_encoder = FingerEncoder(
            out_dim=self.cfg.FINGER_DIM, 
            dropout=self.cfg.DROPOUT
        )

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

        # ── Temporal module ───────────────────────────────────────
        self.temporal = TemporalTransformer(
            model_dim  = cfg.TEMPORAL_DIM,
            num_heads  = cfg.TEMPORAL_HEADS,
            num_layers = cfg.TEMPORAL_LAYERS,
            dropout    = cfg.DROPOUT)

        # ── Temporal attention pooling ────────────────────────────
        self.temporal_pool = nn.Linear(cfg.TEMPORAL_DIM, 1)

        # ── Classifier ────────────────────────────────────────────
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
        """x: (B, T, 208)"""

        # 1. Parse features
        pose, left_xyz, right_xyz, curl_left, curl_right, emotion = \
            FeatureParser.parse(x)

        # 2. Encode từng ngón riêng biệt (xyz landmarks)
        left_fingers  = self.finger_encoder(left_xyz)    # (B,T,6,finger_dim)
        right_fingers = self.finger_encoder(right_xyz)   # (B,T,6,finger_dim)

        # 3. Encode curl features (trạng thái co/duỗi)
        curl_left_emb  = self.curl_encoder(curl_left)    # (B,T,curl_embed_dim)
        curl_right_emb = self.curl_encoder(curl_right)

        # 4. Graph attention: học quan hệ không gian giữa các ngón
        left_fingers  = self.graph_attn_left(left_fingers)
        right_fingers = self.graph_attn_right(right_fingers)

        # 5. Cross-hand attention: học quan hệ 2 tay phối hợp
        left_fingers, right_fingers = self.cross_attn(left_fingers, right_fingers)

        # 6. Aggregate → temporal input
        temporal_in = self.aggregator(
            left_fingers, right_fingers,
            curl_left_emb, curl_right_emb,
            pose, emotion)               # (B,T,temporal_dim)

        # 7. Temporal transformer
        temporal_out = self.temporal(temporal_in)  # (B,T,temporal_dim)

        # 8. Attention pooling: học frame nào quan trọng
        weights = torch.softmax(
            self.temporal_pool(temporal_out).squeeze(-1), dim=-1)   # (B,T)
        context = (temporal_out * weights.unsqueeze(-1)).sum(dim=1)  # (B,D)

        # 9. Classify
        return self.classifier(context)


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
        print(f"\n  Evaluating on test set...")
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
        return self.best_val_acc


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train HandAware VSL Classifier (208 dim — finger curl)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python train.py
  python train.py --epochs 100 --batch_size 16
  python train.py --finger_dim 64 --temporal_layers 6
  python train.py --data_dir data/processed
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
    print()

    train_loader, val_loader, test_loader, label_map = build_dataloaders(cfg)

    model = HandAwareVSLClassifier(num_classes=len(label_map), cfg=cfg)
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

    trainer = Trainer(model, train_loader, val_loader, test_loader,
                      label_map, cfg, ckpt_name=args.ckpt_name)
    trainer.train()
    trainer.evaluate()

    print(f"\n  DONE!")
    print(f"  Checkpoint: {cfg.CHECKPOINT_DIR}/{args.ckpt_name}")
    print(f"  Log       : {cfg.LOG_DIR}/history.json")


if __name__ == "__main__":
    main()