"""
vsl/model.py - DualTransformer model definition
=================================================
    from vsl.model import DualTransformer, load_model
    Soft Gate: mỗi group token tự học attend bao nhiêu
      dựa trên mức độ có data trong group đó
      → interact=0 → gate≈0 → model bỏ qua
      → hand mạnh  → gate≈1 → attend nhiều
"""

import math
import torch
import torch.nn as nn

from vsl.config import cfg


# ═══════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class SpatialTransformer(nn.Module):
    """Xử lý 5 nhóm features theo không gian trong mỗi frame.
    Groups:
      0: pose      (75 dims)
      1: face      (90 dims)
      2: left_hand (63 dims)
      3: right_hand(63 dims)
      4: interact  (55 dims)

    Soft Gate — tự học mức độ attend mỗi group:
      presence = mean(|group_features|)  → đo "group có data không"
      gate     = sigmoid(linear(presence)) → 0~1
      token    = project(group) * gate     → scale down nếu rỗng
    """
    NUM_TOKENS = 5

    def __init__(self, feat_dim, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.d_model = d_model
        # ── FIX: gán vào self.group_dims thay vì biến local ──
        self.group_dims = [
            cfg.POSE_END  - cfg.POSE_START,
            cfg.FACE_END  - cfg.FACE_START,
            63, 63,   # left hand, right hand
            cfg.INTERACT_END - cfg.INTERACT_START,
        ]

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d_model),
            )
            for d in self.group_dims
        ])

        # Token embedding (identity của từng group)
        self.token_embed = nn.Embedding(self.NUM_TOKENS, d_model)

        # Soft Gate — mỗi group 1 linear: scalar → scalar
        self.gate_projs = nn.ModuleList([
            nn.Linear(1, 1) for _ in self.group_dims
        ])
        for gate in self.gate_projs:
            nn.init.constant_(gate.bias, -0.5)

        # Transformer encoder
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _split(self, x: torch.Tensor):
        return [
            x[:, :, cfg.POSE_START   :cfg.POSE_END],
            x[:, :, cfg.FACE_START   :cfg.FACE_END],
            x[:, :, cfg.HAND_START   :cfg.HAND_START + 63],
            x[:, :, cfg.HAND_START+63:cfg.HAND_END],
            x[:, :, cfg.INTERACT_START:cfg.INTERACT_END],
        ]

    def forward(self, x: torch.Tensor,
                return_gates: bool = False):
        B, T, _ = x.shape
        toks      = []
        gate_vals = []

        for i, (g, proj, gate_proj) in enumerate(
                zip(self._split(x), self.group_projs, self.gate_projs)):

            g_flat = g.reshape(B * T, -1)
            presence = g_flat.abs().mean(dim=-1, keepdim=True)
            gate     = torch.sigmoid(gate_proj(presence))
            gate_vals.append(gate.detach().mean().item())

            tok = proj(g_flat) * gate
            tok = tok + self.token_embed.weight[i]
            toks.append(tok.unsqueeze(1))

        tokens = torch.cat(
            [self.cls_token.expand(B * T, -1, -1)] + toks, dim=1)

        out = self.transformer(tokens)[:, 0, :]
        out = out.reshape(B, T, self.d_model)

        if return_gates:
            return out, gate_vals
        return out

    @staticmethod
    def get_gate_names():
        return ['pose', 'face', 'left_hand', 'right_hand', 'interact']


class TemporalTransformer(nn.Module):
    """Xử lý chuỗi thời gian T frames."""
    def __init__(self, d_model, nhead, num_layers, ff_dim, dropout, seq_len):
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model, max_len=seq_len + 1, dropout=dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.pos_enc(
            torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1))
        return self.transformer(x)[:, 0, :]


# ═══════════════════════════════════════════════════════════
# DUAL TRANSFORMER
# ═══════════════════════════════════════════════════════════

class DualTransformer(nn.Module):
    """
    Input  : (B, T, 346)
    Output : (B, num_classes)
    """
    def __init__(self, feat_dim: int, seq_len: int,
                 num_classes: int, config=cfg):
        super().__init__()
        d = config.D_MODEL
        self.spatial = SpatialTransformer(
            feat_dim, d,
            config.SPATIAL_HEADS,  config.SPATIAL_LAYERS,
            config.SPATIAL_FF_DIM, config.SPATIAL_DROPOUT)
        self.temporal = TemporalTransformer(
            d, config.TEMPORAL_HEADS, config.TEMPORAL_LAYERS,
            config.TEMPORAL_FF_DIM, config.TEMPORAL_DROPOUT, seq_len)
        self.classifier = nn.Sequential(
            nn.Linear(d * 2, config.CLASSIFIER_HIDDEN), nn.GELU(),
            nn.Dropout(config.DROPOUT_FINAL),
            nn.Linear(config.CLASSIFIER_HIDDEN,
                      config.CLASSIFIER_HIDDEN // 2), nn.GELU(),
            nn.Dropout(config.DROPOUT_FINAL / 2),
            nn.Linear(config.CLASSIFIER_HIDDEN // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        t = self.temporal(s)
        return self.classifier(torch.cat([s.mean(1), t], dim=-1))

    def forward_with_gates(self, x: torch.Tensor):
        s, gate_vals = self.spatial(x, return_gates=True)
        t            = self.temporal(s)
        logits       = self.classifier(torch.cat([s.mean(1), t], dim=-1))
        gate_dict    = dict(zip(SpatialTransformer.get_gate_names(), gate_vals))
        return logits, gate_dict


# ═══════════════════════════════════════════════════════════
# HELPER: Load checkpoint
# ═══════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = cfg.DEVICE):
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}")

    ckpt      = torch.load(checkpoint_path, map_location=device)
    label_map = ckpt.get('label_map', {})
    if not label_map:
        raise ValueError("Checkpoint khong co label_map!")

    num_classes = len(label_map)
    model = DualTransformer(cfg.FEAT_DIM, cfg.SEQ_LEN, num_classes, cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    epoch   = ckpt.get('epoch', '?')
    val_acc = ckpt.get('val_acc', 0) * 100
    print(f"  Model loaded | classes={num_classes} "
          f"epoch={epoch} val_acc={val_acc:.1f}%")
    return model, label_map, epoch, val_acc