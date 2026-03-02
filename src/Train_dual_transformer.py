"""
train_dual_transformer.py - Huấn luyện VSL Sign Language Recognition
======================================================================
Kiến trúc: Spatial Transformer + Temporal Transformer (DualTransformer)

Chạy:
    python train_dual_transformer.py

Output:
    checkpoints/best_model.pt   ← model tốt nhất (đè file cũ mỗi lần train)
    logs/history_<ts>.json      ← lịch sử loss/acc
    charts/                     ← 13 biểu đồ cho báo cáo nghiên cứu
"""

import os
import json

from vsl.config import cfg as vsl_cfg
from vsl.model  import DualTransformer
from trainer    import build_dataloaders, Trainer


# ══════════════════════════════════════════════════════════
# TRAINING CONFIG
# ══════════════════════════════════════════════════════════

class Config:
    # ── Data ──
    DATA_DIR       = 'data/processed'
    LABEL_MAP_PATH = 'data/processed/label_map.json'
    SEQ_LEN        = vsl_cfg.SEQ_LEN
    FEAT_DIM       = vsl_cfg.FEAT_DIM

    # ── Model (dùng lại từ vsl.config) ──
    D_MODEL           = vsl_cfg.D_MODEL
    SPATIAL_HEADS     = vsl_cfg.SPATIAL_HEADS
    SPATIAL_LAYERS    = vsl_cfg.SPATIAL_LAYERS
    SPATIAL_FF_DIM    = vsl_cfg.SPATIAL_FF_DIM
    SPATIAL_DROPOUT   = vsl_cfg.SPATIAL_DROPOUT
    TEMPORAL_HEADS    = vsl_cfg.TEMPORAL_HEADS
    TEMPORAL_LAYERS   = vsl_cfg.TEMPORAL_LAYERS
    TEMPORAL_FF_DIM   = vsl_cfg.TEMPORAL_FF_DIM
    TEMPORAL_DROPOUT  = vsl_cfg.TEMPORAL_DROPOUT
    CLASSIFIER_HIDDEN = vsl_cfg.CLASSIFIER_HIDDEN
    DROPOUT_FINAL     = vsl_cfg.DROPOUT_FINAL

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

    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def load_label_map(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay: {path}\n"
            f"Chay video_to_npy.py truoc de tao label_map.json!")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(" DUAL TRANSFORMER – VSL TRAINING ".center(60, "="))
    print("=" * 60)
    print(f"\n  Device : {cfg.DEVICE}")
    print(f"  Input  : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")

    label_map   = load_label_map(cfg.LABEL_MAP_PATH)
    num_classes = len(label_map)
    print(f"  Classes: {num_classes} → {list(label_map.keys())}")

    train_loader, val_loader, test_loader, split_counts = \
        build_dataloaders(cfg.DATA_DIR, label_map, cfg)

    model = DualTransformer(
        feat_dim    = cfg.FEAT_DIM,
        seq_len     = cfg.SEQ_LEN,
        num_classes = num_classes,
        config      = cfg,
    )

    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        label_map, cfg, split_counts=split_counts,
    )
    trainer.train()
    trainer.evaluate_and_plot()

    print(f"\n  HOAN THANH!")
    print(f"  Bieu do : {cfg.CHART_DIR}/")
    print(f"  Ckpt    : {cfg.CHECKPOINT_DIR}/best_model.pt\n")


if __name__ == '__main__':
    main()