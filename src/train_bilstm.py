"""
train_bilstm.py - Huấn luyện VSL Sign Language Recognition
===========================================================
Kiến trúc: Bidirectional LSTM + Attention

Chạy:
    python -m src.lstm.train_bilstm

Output:
    checkpoints/bilstm_best.pt  ← model tốt nhất (tên mới mỗi lần train)
    logs/bilstm_history_<ts>.json           ← lịch sử loss/acc
    charts/bilstm_*                         ← biểu đồ training + confusion matrix + attention
"""

import os
import json
import math
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

from vsl.config import cfg as vsl_cfg
from vsl.config_lite import cfg as vsl_cfg

from visualize_bilstm import BiLSTMVisualizer
from visualize_bilstm import BiLSTMVisualizer


# ══════════════════════════════════════════════════════════
# TRAINING CONFIG
# ══════════════════════════════════════════════════════════

class Config:
    # ── Data ──
    DATA_DIR       = 'data/processed_lite'
    LABEL_MAP_PATH = 'data/processed/label_map.json'
    SEQ_LEN        = vsl_cfg.SEQ_LEN        # 64
    FEAT_DIM       = vsl_cfg.FEAT_DIM       # 346

    # ── BiLSTM Architecture ──
    HIDDEN_DIM     = 256     # hidden size mỗi hướng
    NUM_LAYERS     = 3       # số lớp LSTM
    DROPOUT_LSTM   = 0.3     # dropout giữa các lớp LSTM
    DROPOUT_FC     = 0.4     # dropout trước classifier
    BIDIRECTIONAL  = True
    USE_ATTENTION  = True    # Attention mechanism

    # ── Training ──
    EPOCHS       = 100
    BATCH_SIZE   = 32
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 15        # early stopping
    GRAD_CLIP    = 1.0

    # ── LR Scheduler ──
    LR_SCHEDULER     = 'cosine'   # 'cosine' | 'step' | 'plateau'
    LR_WARMUP_EPOCHS = 5
    LR_MIN           = 1e-6
    LR_STEP_SIZE     = 20
    LR_GAMMA         = 0.5

    # ── Output ──
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR        = 'logs'
    CHART_DIR      = 'charts'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


# ══════════════════════════════════════════════════════════
# DATASET (giữ nguyên như Dual Transformer)
# ══════════════════════════════════════════════════════════

class VSLDataset(Dataset):
    def __init__(self, data_dir, label_map, split='train', augment=False):
        self.samples = []
        self.augment = augment
        split_dir = os.path.join(data_dir, split)

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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)
        if self.augment:
            data = self._runtime_aug(data)
        return torch.from_numpy(data), label

    def _runtime_aug(self, data):
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


def build_dataloaders(data_dir, label_map, cfg):
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
            "  Chay video_to_npy.py truoc de tao file .npy")

    kw = dict(num_workers=0, pin_memory=(cfg.DEVICE == 'cuda'))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)

    # split counts
    idx2label = {v: k for k, v in label_map.items()}
    counts = {name: {'train': 0, 'val': 0, 'test': 0} for name in label_map}
    for ds, sname in [(train_ds,'train'), (val_ds,'val'), (test_ds,'test')]:
        for _, li in ds.samples:
            n = idx2label.get(li, str(li))
            if n in counts: counts[n][sname] += 1

    return train_loader, val_loader, test_loader, counts


# ══════════════════════════════════════════════════════════
# MODEL: Bidirectional LSTM + Attention
# ══════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    """Bahdanau-style attention trên output của BiLSTM."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)        # (B, T)
        weights = torch.softmax(scores, dim=-1)         # (B, T)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        return context, weights


class BiLSTMClassifier(nn.Module):
    """
    Input  : (B, T, FEAT_DIM)
    Output : (B, num_classes)

    Luồng:
      input_proj → LayerNorm
        → BiLSTM (N layers)
        → Attention (optional)
        → concat(attn_ctx, last_hidden)
        → FC classifier
    """
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.3, dropout_fc=0.4,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_dirs      = 2 if bidirectional else 1

        # Input projection + norm
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_dirs

        # Attention
        if use_attention:
            self.attention = AttentionLayer(lstm_out_dim)
            fc_in = lstm_out_dim * 2   # attn_ctx + last_hidden
        else:
            fc_in = lstm_out_dim

        # Classifier
        mid = max(num_classes * 4, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, mid), nn.GELU(),
            nn.Dropout(dropout_fc),
            nn.Linear(mid, mid // 2), nn.GELU(),
            nn.Dropout(dropout_fc / 2),
            nn.Linear(mid // 2, num_classes),
        )

    def forward(self, x, return_attention=False):
        # x: (B, T, FEAT_DIM)
        x = self.input_proj(x)              # (B, T, hidden_dim)
        lstm_out, (hn, _) = self.lstm(x)    # lstm_out: (B, T, hidden*dirs)

        # Last hidden from both directions
        if self.bidirectional:
            last_h = torch.cat([hn[-2], hn[-1]], dim=-1)  # (B, hidden*2)
        else:
            last_h = hn[-1]                                # (B, hidden)

        attn_weights = None
        if self.use_attention:
            ctx, attn_weights = self.attention(lstm_out)   # (B, hidden*dirs)
            feat = torch.cat([ctx, last_h], dim=-1)        # (B, hidden*dirs*2)
        else:
            feat = last_h

        logits = self.classifier(feat)

        if return_attention:
            return logits, attn_weights
        return logits


# ══════════════════════════════════════════════════════════
# LR SCHEDULER BUILDER
# ══════════════════════════════════════════════════════════

def build_scheduler(optimizer, cfg, total_steps_per_epoch):
    if cfg.LR_SCHEDULER == 'cosine':
        # Warmup + cosine decay
        def lr_lambda(epoch):
            if epoch < cfg.LR_WARMUP_EPOCHS:
                return (epoch + 1) / cfg.LR_WARMUP_EPOCHS
            progress = (epoch - cfg.LR_WARMUP_EPOCHS) / max(1, cfg.EPOCHS - cfg.LR_WARMUP_EPOCHS)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return cfg.LR_MIN / cfg.LR + (1 - cfg.LR_MIN / cfg.LR) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif cfg.LR_SCHEDULER == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    elif cfg.LR_SCHEDULER == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, min_lr=cfg.LR_MIN)

    return None


# ══════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════

class BiLSTMTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 label_map, cfg, split_counts=None):
        self.model        = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.label_map    = label_map
        self.idx2label    = {v: k for k, v in label_map.items()}
        self.cfg          = cfg
        self.split_counts = split_counts or {}

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = build_scheduler(self.optimizer, cfg, len(train_loader))
        self.scaler    = torch.cuda.amp.GradScaler() if cfg.DEVICE == 'cuda' else None

        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
            'lr':         [],
        }
        self.best_val_acc = 0.0
        self.patience_cnt = 0


        # Visualizer — tất cả chart lưu với prefix bilstm_
        self.viz = BiLSTMVisualizer(label_map, output_dir=cfg.CHART_DIR)

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)
        os.makedirs(cfg.CHART_DIR,      exist_ok=True)

        # Visualizer (sinh chart bilstm_01_... không đụng file DualTransformer)
        self.viz = BiLSTMVisualizer(label_map, output_dir=cfg.CHART_DIR)

    # ── Single epoch ──────────────────────────────────────

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.cfg.DEVICE)
                y = y.to(self.cfg.DEVICE)

                if self.scaler and train:
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss   = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.GRAD_CLIP)
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
                preds      = logits.argmax(dim=-1)
                correct    += (preds == y).sum().item()
                total      += y.size(0)

        return total_loss / total, correct / total

    # ── Train loop ────────────────────────────────────────

    def train(self):
        print(f"\n  Bat dau training BiLSTM...")
        print(f"  Scheduler: {self.cfg.LR_SCHEDULER.upper()}")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader,   train=False)

            # Update scheduler
            cur_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl_acc)
                else:
                    self.scheduler.step()

            # Save history
            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)
            self.history['lr'].append(cur_lr)

            # Update visualizer
            self.viz.update(epoch, tr_loss, vl_loss, tr_acc, vl_acc, cur_lr)

            # Cập nhật visualizer sau mỗi epoch
            self.viz.update(epoch, tr_loss, vl_loss, tr_acc, vl_acc, cur_lr)

            elapsed = time.time() - t0
            print(f"  Ep {epoch:03d}/{self.cfg.EPOCHS} | "
                  f"TrLoss={tr_loss:.4f} TrAcc={tr_acc*100:.1f}% | "
                  f"VlLoss={vl_loss:.4f} VlAcc={vl_acc*100:.1f}% | "
                  f"LR={cur_lr:.2e} | {elapsed:.1f}s")

            # Checkpoint save (tên mới mỗi lần train)
            if vl_acc > self.best_val_acc:
                self.best_val_acc = vl_acc
                self.patience_cnt = 0
                ckpt_path = os.path.join(
                    self.cfg.CHECKPOINT_DIR,
                    f'bilstm_best.pt')
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer':   self.optimizer.state_dict(),
                    'val_acc':     vl_acc,
                    'val_loss':    vl_loss,
                    'label_map':   self.label_map,
                    'cfg': {
                        'FEAT_DIM':   self.cfg.FEAT_DIM,
                        'SEQ_LEN':    self.cfg.SEQ_LEN,
                        'HIDDEN_DIM': self.cfg.HIDDEN_DIM,
                        'NUM_LAYERS': self.cfg.NUM_LAYERS,
                    },
                }, ckpt_path)
                print(f"  ✓ Checkpoint saved → {ckpt_path} (val_acc={vl_acc*100:.2f}%)")
            else:
                self.patience_cnt += 1

            # Early stopping
            if self.patience_cnt >= self.cfg.PATIENCE:
                print(f"\n  Early stopping tại epoch {epoch} "
                      f"(patience={self.cfg.PATIENCE})")
                break

        # Save history JSON
        log_path = os.path.join(self.cfg.LOG_DIR, f'bilstm_history.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"  Log saved → {log_path}")

    # ══════════════════════════════════════════════════════
    # EVALUATE + CHARTS
    # ══════════════════════════════════════════════════════

    def evaluate_and_plot(self):
        # ── Classification report (text) ──
        print("\n  Evaluating on test set...")
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.cfg.DEVICE)
                logits = self.model(x)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(y.tolist())
        from sklearn.metrics import classification_report as _clsrpt
        class_names = [self.idx2label[i] for i in range(len(self.label_map))]
        report = _clsrpt(all_labels, all_preds, target_names=class_names, digits=4)
        print("\n" + "="*60 + "\nCLASSIFICATION REPORT\n" + "="*60)
        print(report)
        rpt_path = os.path.join(self.cfg.LOG_DIR, f'bilstm_report.txt')
        with open(rpt_path, 'w', encoding='utf-8') as rf:
            rf.write(report)
        print(f"  Report -> {rpt_path}")
        # ── 13 biểu đồ qua BiLSTMVisualizer ──
        self.viz.plot_all(
            model        = self.model,
            test_loader  = self.test_loader,
            device       = self.cfg.DEVICE,
            cfg          = self.cfg,
            split_counts = self.split_counts,
        )
        print(f"  Best val acc : {self.best_val_acc*100:.2f}%")


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


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(" BIDIRECTIONAL LSTM – VSL TRAINING ".center(60, "="))
    print("=" * 60)
    print(f"\n  Device    : {cfg.DEVICE}")
    print(f"  Input     : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")
    print(f"  Hidden    : {cfg.HIDDEN_DIM} × {cfg.NUM_LAYERS} layers (bidirectional={cfg.BIDIRECTIONAL})")
    print(f"  Attention : {cfg.USE_ATTENTION}")
    print(f"  Scheduler : {cfg.LR_SCHEDULER.upper()}")

    label_map   = load_label_map(cfg.LABEL_MAP_PATH)
    num_classes = len(label_map)
    print(f"  Classes   : {num_classes} → {list(label_map.keys())}")

    train_loader, val_loader, test_loader, split_counts = \
        build_dataloaders(cfg.DATA_DIR, label_map, cfg)

    model = BiLSTMClassifier(
        feat_dim      = cfg.FEAT_DIM,
        hidden_dim    = cfg.HIDDEN_DIM,
        num_layers    = cfg.NUM_LAYERS,
        num_classes   = num_classes,
        dropout_lstm  = cfg.DROPOUT_LSTM,
        dropout_fc    = cfg.DROPOUT_FC,
        bidirectional = cfg.BIDIRECTIONAL,
        use_attention = cfg.USE_ATTENTION,
    )

    total, trainable = count_parameters(model)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable\n")

    trainer = BiLSTMTrainer(
        model, train_loader, val_loader, test_loader,
        label_map, cfg, split_counts=split_counts,
    )
    trainer.train()
    trainer.evaluate_and_plot()

    print(f"\n  HOAN THANH!")
    print(f"  Charts  : {cfg.CHART_DIR}/bilstm_*/")
    print(f"  Ckpt    : {cfg.CHECKPOINT_DIR}/bilstm_best.pt\n")


if __name__ == '__main__':
    main()