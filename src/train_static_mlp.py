"""
train_static_mlp.py - Train Static MLP cho fingerspelling VSL
=============================================================
Input : data/static/ (sinh ra từ video_to_npy_static.py)
        mỗi file .npy có shape (96,)

Output:
    checkpoints/static_mlp_best_<timestamp>.pt
    logs/static_history_<timestamp>.json
    charts/static_*

Chạy:
    python src/static/train_static_mlp.py
"""

import os
import sys
import json
import time
import math
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ── Fix sys.path ──────────────────────────────────────────────────
# File ở src/train_static_mlp.py → parents[1] = vsl-recognition-project/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

class Config:
    # ── Data ──
    DATA_DIR       = str(_PROJECT_ROOT / 'datamlp' / 'static')
    LABEL_MAP_PATH = str(_PROJECT_ROOT / 'datamlp' / 'static' / 'label_map.json')
    FEAT_DIM       = 96

    # ── Model ──
    HIDDEN_1   = 256
    HIDDEN_2   = 128
    HIDDEN_3   = 64
    DROPOUT_1  = 0.4
    DROPOUT_2  = 0.3
    DROPOUT_3  = 0.2

    # ── Training ──
    EPOCHS       = 150
    BATCH_SIZE   = 64
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 20
    GRAD_CLIP    = 1.0

    # ── LR Scheduler ──
    LR_WARMUP    = 5
    LR_MIN       = 1e-6

    # ── Augmentation ──
    AUGMENT_NOISE = 0.005   # Gaussian noise nhỏ khi train

    # ── Output ──
    CHECKPOINT_DIR = str(_PROJECT_ROOT / 'checkpoints')
    LOG_DIR        = str(_PROJECT_ROOT / 'logs')
    CHART_DIR      = str(_PROJECT_ROOT / 'charts')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class StaticDataset(Dataset):
    """
    Đọc file .npy (96,) từ data/static/<split>/<label>/*.npy
    """
    def __init__(self, data_dir: str, label_map: dict,
                 split: str = 'train', augment: bool = False):
        self.samples = []
        self.augment = augment
        split_dir    = Path(data_dir) / split

        if not split_dir.is_dir():
            print(f"  [WARN] Khong tim thay: {split_dir}")
            return

        for label_name, label_idx in label_map.items():
            label_dir = split_dir / label_name
            if not label_dir.is_dir():
                continue
            npy_files = sorted(label_dir.glob('*.npy'))
            for fp in npy_files:
                self.samples.append((str(fp), label_idx))

        print(f"  [{split:5s}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path).astype(np.float32)   # (96,)

        if self.augment:
            # Gaussian noise nhỏ → giả lập tay run nhẹ
            noise = np.random.normal(0, cfg.AUGMENT_NOISE,
                                     feat.shape).astype(np.float32)
            feat  = feat + noise

        return torch.from_numpy(feat), label


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class StaticMLP(nn.Module):
    """
    MLP 3 lớp ẩn cho fingerspelling static.

    Input : (B, 96)
    Output: (B, num_classes)

    Kiến trúc:
      96 → LayerNorm → 256 → GELU → Dropout(0.4)
         → 128 → GELU → Dropout(0.3)
         → 64  → GELU → Dropout(0.2)
         → num_classes
    """
    def __init__(self, feat_dim: int, num_classes: int, cfg=cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),

            nn.Linear(feat_dim, cfg.HIDDEN_1),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_1),

            nn.Linear(cfg.HIDDEN_1, cfg.HIDDEN_2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_2),

            nn.Linear(cfg.HIDDEN_2, cfg.HIDDEN_3),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_3),

            nn.Linear(cfg.HIDDEN_3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
# LR SCHEDULER (cosine warmup)
# ══════════════════════════════════════════════════════════════════

def build_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < cfg.LR_WARMUP:
            return (epoch + 1) / cfg.LR_WARMUP
        progress = (epoch - cfg.LR_WARMUP) / max(1, cfg.EPOCHS - cfg.LR_WARMUP)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.LR_MIN / cfg.LR + (1 - cfg.LR_MIN / cfg.LR) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════

class StaticTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 label_map, cfg):
        self.model       = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.label_map    = label_map
        self.idx2label    = {v: k for k, v in label_map.items()}
        self.cfg          = cfg

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = build_scheduler(self.optimizer)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
            'lr':         [],
        }
        self.best_val_acc = 0.0
        self.patience_cnt = 0
        self.ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)
        os.makedirs(cfg.CHART_DIR,      exist_ok=True)

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.cfg.DEVICE)
                y = y.to(self.cfg.DEVICE)

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
        print(f"\n  Bat dau training Static MLP...")
        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader,   train=False)

            cur_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)
            self.history['lr'].append(cur_lr)

            print(f"  Ep {epoch:03d}/{self.cfg.EPOCHS} | "
                  f"TrLoss={tr_loss:.4f} TrAcc={tr_acc*100:.1f}% | "
                  f"VlLoss={vl_loss:.4f} VlAcc={vl_acc*100:.1f}% | "
                  f"LR={cur_lr:.2e} | {time.time()-t0:.1f}s")

            # Checkpoint
            if vl_acc > self.best_val_acc:
                self.best_val_acc = vl_acc
                self.patience_cnt = 0
                ckpt_path = os.path.join(
                    self.cfg.CHECKPOINT_DIR,
                    f'static_mlp_best_{self.ts}.pt')
                torch.save({
                    'epoch'      : epoch,
                    'model_state': self.model.state_dict(),
                    'val_acc'    : vl_acc,
                    'label_map'  : self.label_map,
                    'cfg': {
                        'FEAT_DIM'  : self.cfg.FEAT_DIM,
                        'HIDDEN_1'  : self.cfg.HIDDEN_1,
                        'HIDDEN_2'  : self.cfg.HIDDEN_2,
                        'HIDDEN_3'  : self.cfg.HIDDEN_3,
                    },
                }, ckpt_path)
                print(f"  ✓ Checkpoint → {Path(ckpt_path).name} "
                      f"(val_acc={vl_acc*100:.2f}%)")
            else:
                self.patience_cnt += 1

            if self.patience_cnt >= self.cfg.PATIENCE:
                print(f"\n  Early stopping tại epoch {epoch}")
                break

        # Save log
        log_path = os.path.join(
            self.cfg.LOG_DIR, f'static_history_{self.ts}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"  Log → {log_path}")

    def evaluate_and_plot(self):
        print("\n  Evaluating on test set...")
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x      = x.to(self.cfg.DEVICE)
                preds  = self.model(x).argmax(-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())

        class_names = [self.idx2label[i]
                       for i in range(len(self.label_map))]

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT – Static MLP")
        print("=" * 60)
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names, digits=4)
        print(report)

        # Save report
        rpt = os.path.join(
            self.cfg.LOG_DIR, f'static_report_{self.ts}.txt')
        with open(rpt, 'w', encoding='utf-8') as f:
            f.write(report)

        self._plot_all(all_labels, all_preds, class_names)
        print(f"\n  Best val acc : {self.best_val_acc*100:.2f}%")

    # ── Charts ────────────────────────────────────────────────────

    def _save(self, name):
        path = os.path.join(
            self.cfg.CHART_DIR, f'static_{name}_{self.ts}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [Chart] {path}")

    def _plot_all(self, y_true, y_pred, class_names):
        self._plot_curves()
        self._plot_confusion(y_true, y_pred, class_names)
        self._plot_f1(y_true, y_pred, class_names)
        self._plot_lr()

    def _plot_curves(self):
        ep = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Static MLP – Training History',
                     fontsize=13, fontweight='bold')

        # Loss
        axes[0].plot(ep, self.history['train_loss'],
                     color='#2E86AB', lw=2, label='Train')
        axes[0].plot(ep, self.history['val_loss'],
                     color='#E84855', lw=2, label='Val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        tr_acc = [a*100 for a in self.history['train_acc']]
        vl_acc = [a*100 for a in self.history['val_acc']]
        axes[1].plot(ep, tr_acc, color='#2E86AB', lw=2, label='Train')
        axes[1].plot(ep, vl_acc, color='#E84855', lw=2, label='Val')
        axes[1].axhline(self.best_val_acc*100, color='green',
                        ls='--', alpha=0.7,
                        label=f'Best {self.best_val_acc*100:.1f}%')
        axes[1].set_title('Accuracy'); axes[1].legend()
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Acc (%)')
        axes[1].set_ylim(0, 105); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save('training_curves')

    def _plot_confusion(self, y_true, y_pred, class_names):
        cm   = confusion_matrix(y_true, y_pred)
        n    = len(class_names)
        size = max(8, n * 0.7)

        fig, axes = plt.subplots(1, 2, figsize=(size*2, size))
        fig.suptitle('Static MLP – Confusion Matrix',
                     fontsize=13, fontweight='bold')

        for ax, norm, title in zip(
                axes, [True, False],
                ['Normalized (%)', 'Raw Count']):
            if norm:
                with np.errstate(divide='ignore', invalid='ignore'):
                    data = np.where(
                        cm.sum(1, keepdims=True) == 0, 0,
                        cm / cm.sum(1, keepdims=True) * 100)
                fmt  = '.1f'
                vmax = 100
            else:
                data = cm; fmt = 'd'; vmax = cm.max()

            im = ax.imshow(data, cmap='Blues', vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(class_names, rotation=45,
                               ha='right', fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(title, fontweight='bold')

            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{data[i,j]:{fmt}}',
                            ha='center', va='center', fontsize=7,
                            color='white' if data[i,j] > thresh else 'black')

        plt.tight_layout()
        self._save('confusion_matrix')

    def _plot_f1(self, y_true, y_pred, class_names):
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(len(class_names))),
            zero_division=0)

        n = len(class_names)
        x = np.arange(n)
        w = 0.26

        fig, ax = plt.subplots(figsize=(max(10, n*0.8), 6))
        ax.bar(x - w, prec*100, w, label='Precision',
               color='#5E81AC', alpha=0.85)
        ax.bar(x,     rec*100,  w, label='Recall',
               color='#A3BE8C', alpha=0.85)
        ax.bar(x + w, f1*100,   w, label='F1-Score',
               color='#BF616A', alpha=0.85)
        ax.axhline(f1.mean()*100, color='black', ls='--', lw=1.5,
                   label=f'Macro F1: {f1.mean()*100:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Score (%)')
        ax.set_title('Static MLP – Per-Class Precision / Recall / F1',
                     fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        self._save('precision_recall_f1')

    def _plot_lr(self):
        ep = range(1, len(self.history['lr']) + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(ep, self.history['lr'], color='#F4A261', lw=2)
        plt.fill_between(ep, self.history['lr'], alpha=0.15, color='#F4A261')
        plt.title('Static MLP – Learning Rate Schedule',
                  fontsize=13, fontweight='bold')
        plt.xlabel('Epoch'); plt.ylabel('LR')
        plt.yscale('log'); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('lr_schedule')


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(" STATIC MLP – VSL FINGERSPELLING TRAINING ".center(60, "="))
    print("=" * 60)
    print(f"\n  Device   : {cfg.DEVICE}")
    print(f"  Feat dim : {cfg.FEAT_DIM}")
    print(f"  Data dir : {cfg.DATA_DIR}")

    # Load label map
    if not os.path.exists(cfg.LABEL_MAP_PATH):
        print(f"\n  [ERROR] Khong tim thay label_map.json")
        print(f"  Chay truoc: python src/static/video_to_npy_static.py")
        sys.exit(1)

    with open(cfg.LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    print(f"  Classes  : {num_classes} → {list(label_map.keys())}")

    # Datasets
    train_ds = StaticDataset(cfg.DATA_DIR, label_map,
                              split='train', augment=True)
    val_ds   = StaticDataset(cfg.DATA_DIR, label_map,
                              split='val',   augment=False)
    test_ds  = StaticDataset(cfg.DATA_DIR, label_map,
                              split='test',  augment=False)

    if len(train_ds) == 0:
        print("\n  [ERROR] Train dataset rong!")
        print(f"  Kiem tra: {cfg.DATA_DIR}/train/")
        sys.exit(1)

    kw = dict(num_workers=0, pin_memory=(cfg.DEVICE == 'cuda'))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)

    # Model
    model = StaticMLP(feat_dim=cfg.FEAT_DIM, num_classes=num_classes)
    print(f"  Params   : {model.count_params():,}")

    # Train
    trainer = StaticTrainer(model, train_loader, val_loader,
                             test_loader, label_map, cfg)
    trainer.train()
    trainer.evaluate_and_plot()

    print(f"\n  HOAN THANH!")
    print(f"  Ckpt  : checkpoints/static_mlp_best_<timestamp>.pt")
    print(f"  Charts: charts/static_*/\n")


if __name__ == '__main__':
    main()