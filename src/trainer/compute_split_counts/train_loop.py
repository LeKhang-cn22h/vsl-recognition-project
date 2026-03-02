"""
trainer/train_loop.py - Training loop + Evaluate
=================================================
    from trainer.train_loop import Trainer
"""

import os
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report

from visualize_training import Visualizer


class Trainer:
    """
    Vòng lặp train: forward → backward → scheduler → early stopping.
    Tự lưu best_model.pt và history log.
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 label_map: dict, cfg, split_counts=None):

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
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)

        self.best_val_acc = 0.0
        self.patience_cnt = 0

        # Checkpoint: cố định tên → đè file cũ khi train lại
        self.ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pt')
        # Log: giữ timestamp để phân biệt lịch sử các lần train
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path  = os.path.join(cfg.LOG_DIR, f'history_{ts}.json')

        self.viz = Visualizer(label_map, output_dir=cfg.CHART_DIR)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Model   : DualTransformer | Params: {n_params:,}")
        print(f"  Device  : {self.device}")
        print(f"  Train/Val/Test: "
              f"{len(train_loader.dataset)}/"
              f"{len(val_loader.dataset)}/"
              f"{len(test_loader.dataset)}")
        print(f"  Ckpt    : {self.ckpt_path}\n")

    # ── 1 epoch ──────────────────────────────────────────

    def _run_epoch(self, loader, train: bool = True) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y   = X.to(self.device), y.to(self.device)
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

    # ── Training loop ────────────────────────────────────

    def train(self):
        print("=" * 60)
        print(" BAT DAU TRAINING ".center(60))
        print("=" * 60)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            tl, ta = self._run_epoch(self.train_loader, train=True)
            vl, va = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            self.viz.update(epoch, tl, vl, ta, va, lr)

            print(f"  Epoch {epoch:3d}/{self.cfg.EPOCHS} | "
                  f"Loss {tl:.4f}/{vl:.4f} | "
                  f"Acc {ta*100:5.1f}%/{va*100:5.1f}% | "
                  f"LR {lr:.2e}")

            if va > self.best_val_acc:
                self.best_val_acc = va
                self.patience_cnt = 0
                torch.save({
                    'epoch':           epoch,
                    'model_state':     self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_acc':         va,
                    'label_map':       self.label_map,
                }, self.ckpt_path)
                print(f"  ✓ Best saved (val={va*100:.1f}%)")
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= self.cfg.PATIENCE:
                    print(f"\n  Early stopping @ epoch {epoch}")
                    break

        with open(self.log_path, 'w') as f:
            json.dump(self.viz.history, f, indent=2)
        print(f"\n  History: {self.log_path}")

    # ── Evaluate + plot ──────────────────────────────────

    def evaluate_and_plot(self) -> float:
        print("\n" + "=" * 60)
        print(" EVALUATE + XUAT BIEU DO ".center(60))
        print("=" * 60)

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])

        self.viz.plot_all(
            model        = self.model,
            test_loader  = self.test_loader,
            device       = self.device,
            cfg          = self.cfg,
            split_counts = self.split_counts,
        )

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                preds = self.model(X.to(self.device)).argmax(-1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())

        idx2label = {v: k for k, v in self.label_map.items()}
        names     = [idx2label[i] for i in range(len(self.label_map))]
        test_acc  = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
        print(f"  Best Val Acc  : {self.best_val_acc*100:.2f}%")
        print("\n  Classification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=names, zero_division=0))
        return test_acc