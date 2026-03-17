"""
TRAINING VISUALIZATION - VSL Dual Transformer
==============================================
Xuất đầy đủ biểu đồ chất lượng cao cho báo cáo nghiên cứu khoa học

Biểu đồ được tạo:
  1.  Loss Curve          – Train vs Validation loss theo epoch
  2.  Accuracy Curve      – Train vs Validation accuracy theo epoch
  3.  Learning Rate       – LR schedule theo epoch
  4.  Loss + Acc (combo)  – Gộp 2 biểu đồ trên vào 1 figure
  5.  Confusion Matrix    – Heatmap chuẩn hoá + raw count
  6.  Per-class F1        – Bar chart F1-score từng nhãn
  7.  Precision/Recall    – Grouped bar chart từng lớp
  8.  ROC Curve           – One-vs-rest cho từng class (nếu có proba)
  9.  t-SNE               – Phân bố feature embedding cuối
  10. Grad-CAM Temporal   – Attention weight trên trục thời gian
  11. Model Architecture  – Sơ đồ khối kiến trúc Dual Transformer
  12. Dataset Distribution – Phân bố số mẫu mỗi nhãn (train/val/test)
  13. Training Summary     – 1 trang tóm tắt toàn bộ kết quả

Tất cả ảnh lưu vào: charts/
  - Định dạng: PNG 300 DPI (in được, đưa vào Word/LaTeX)
  - Font: Times New Roman (phù hợp báo cáo khoa học)

Cách dùng (tích hợp vào train_dual_transformer.py):
    from visualize_training import Visualizer
    viz = Visualizer(label_map, output_dir='charts')
    # Gọi sau mỗi epoch:
    viz.update(epoch, train_loss, val_loss, train_acc, val_acc, lr)
    # Gọi sau khi train xong:
    viz.plot_all(model, test_loader, device)
"""

import os
import json
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (không cần display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# ── Cấu hình font cho báo cáo khoa học ──
rcParams.update({
    'font.family'       : 'DejaVu Serif',  # fallback nếu không có Times
    'font.size'         : 11,
    'axes.titlesize'    : 13,
    'axes.labelsize'    : 11,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'legend.fontsize'   : 10,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
})

# ── Bảng màu nhất quán ──
C_TRAIN  = '#2E86AB'   # xanh dương
C_VAL    = '#E84855'   # đỏ
C_TEST   = '#3BB273'   # xanh lá
C_LR     = '#F4A261'   # cam
CMAP_CM  = LinearSegmentedColormap.from_list(
    'cm_blue', ['#FFFFFF', '#2E86AB', '#1A3A4A'])


# ═══════════════════════════════════════════════════════════════════
class Visualizer:
    """
    Thu thập metrics trong quá trình train và xuất biểu đồ.
    """

    def __init__(self, label_map: dict, output_dir: str = 'charts'):
        self.label_map  = label_map
        self.idx2label  = {v: k for k, v in label_map.items()}
        self.labels_list = [self.idx2label[i] for i in range(len(label_map))]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Lịch sử training
        self.history = dict(
            epoch=[],
            train_loss=[], val_loss=[],
            train_acc=[],  val_acc=[],
            lr=[],
        )

        print(f"  [Visualizer] Charts se luu vao: {output_dir}/")

    # ── Cập nhật sau mỗi epoch ──────────────────────────────────────
    def update(self, epoch, train_loss, val_loss,
               train_acc, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc * 100)
        self.history['val_acc'].append(val_acc * 100)
        self.history['lr'].append(lr)

    # ── Lưu ảnh helper ──────────────────────────────────────────────
    def _save(self, fig, name, dpi=300):
        path = os.path.join(self.output_dir, f'{name}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [Chart] Da luu: {path}")
        return path

    # ════════════════════════════════════════════════════════════════
    # 1. LOSS CURVE
    # ════════════════════════════════════════════════════════════════
    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = self.history['epoch']
        ax.plot(ep, self.history['train_loss'],
                color=C_TRAIN, lw=2, label='Train Loss', marker='o',
                markersize=3, markevery=max(1, len(ep)//15))
        ax.plot(ep, self.history['val_loss'],
                color=C_VAL, lw=2, label='Validation Loss', marker='s',
                markersize=3, markevery=max(1, len(ep)//15))

        # Đánh dấu điểm val loss thấp nhất
        best_ep = ep[int(np.argmin(self.history['val_loss']))]
        best_vl = min(self.history['val_loss'])
        ax.axvline(best_ep, color=C_VAL, lw=1, ls=':', alpha=0.7)
        ax.annotate(f'Best\nEpoch {best_ep}', xy=(best_ep, best_vl),
                    xytext=(best_ep + max(1, len(ep)*0.05), best_vl),
                    fontsize=9, color=C_VAL,
                    arrowprops=dict(arrowstyle='->', color=C_VAL, lw=1.2))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Cross-Entropy)')
        ax.set_title('Training & Validation Loss', fontweight='bold')
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '01_loss_curve')

    # ════════════════════════════════════════════════════════════════
    # 2. ACCURACY CURVE
    # ════════════════════════════════════════════════════════════════
    def plot_accuracy(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = self.history['epoch']
        ax.plot(ep, self.history['train_acc'],
                color=C_TRAIN, lw=2, label='Train Accuracy',
                marker='o', markersize=3, markevery=max(1, len(ep)//15))
        ax.plot(ep, self.history['val_acc'],
                color=C_VAL, lw=2, label='Validation Accuracy',
                marker='s', markersize=3, markevery=max(1, len(ep)//15))

        best_ep  = ep[int(np.argmax(self.history['val_acc']))]
        best_acc = max(self.history['val_acc'])
        ax.axvline(best_ep, color=C_VAL, lw=1, ls=':', alpha=0.7)
        ax.annotate(f'{best_acc:.1f}%\nEpoch {best_ep}',
                    xy=(best_ep, best_acc),
                    xytext=(best_ep + max(1, len(ep)*0.05), best_acc - 5),
                    fontsize=9, color=C_VAL,
                    arrowprops=dict(arrowstyle='->', color=C_VAL, lw=1.2))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training & Validation Accuracy', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '02_accuracy_curve')

    # ════════════════════════════════════════════════════════════════
    # 3. LEARNING RATE SCHEDULE
    # ════════════════════════════════════════════════════════════════
    def plot_lr(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        ep = self.history['epoch']
        ax.plot(ep, self.history['lr'],
                color=C_LR, lw=2, label='Learning Rate')
        ax.fill_between(ep, self.history['lr'],
                         alpha=0.15, color=C_LR)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (Cosine Annealing)',
                     fontweight='bold')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '03_lr_schedule')

    # ════════════════════════════════════════════════════════════════
    # 4. COMBO: LOSS + ACCURACY (phù hợp đưa vào báo cáo 1 ảnh)
    # ════════════════════════════════════════════════════════════════
    def plot_training_combo(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ep = self.history['epoch']

        # Loss
        ax1.plot(ep, self.history['train_loss'], color=C_TRAIN,
                 lw=2, label='Train', marker='o', markersize=2,
                 markevery=max(1, len(ep)//15))
        ax1.plot(ep, self.history['val_loss'], color=C_VAL,
                 lw=2, label='Validation', marker='s', markersize=2,
                 markevery=max(1, len(ep)//15))
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('(a) Loss Curve', fontweight='bold')
        ax1.legend(framealpha=0.9)

        # Accuracy
        ax2.plot(ep, self.history['train_acc'], color=C_TRAIN,
                 lw=2, label='Train', marker='o', markersize=2,
                 markevery=max(1, len(ep)//15))
        ax2.plot(ep, self.history['val_acc'], color=C_VAL,
                 lw=2, label='Validation', marker='s', markersize=2,
                 markevery=max(1, len(ep)//15))
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('(b) Accuracy Curve', fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.legend(framealpha=0.9)

        fig.suptitle('Dual Transformer – Training History',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self._save(fig, '04_training_combo')

    # ════════════════════════════════════════════════════════════════
    # 5. CONFUSION MATRIX
    # ════════════════════════════════════════════════════════════════
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        cm = confusion_matrix(y_true, y_pred)
        n  = len(self.labels_list)

        # Tính số cột tốt nhất để chia nhãn dài
        tick_labels = [l.replace('_', '\n') for l in self.labels_list]

        fig_size = max(8, n * 0.85)
        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2 + 1, fig_size))

        for idx, (ax, norm, title) in enumerate(zip(
                axes,
                [True, False],
                ['(a) Normalized (%)', '(b) Raw Count'])):

            if norm:
                with np.errstate(divide='ignore', invalid='ignore'):
                    data = np.where(cm.sum(axis=1, keepdims=True) == 0, 0,
                                    cm / cm.sum(axis=1, keepdims=True) * 100)
                fmt  = '.1f'
                vmax = 100
            else:
                data = cm
                fmt  = 'd'
                vmax = cm.max()

            im = ax.imshow(data, interpolation='nearest',
                           cmap=CMAP_CM, vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlabel('Predicted Label', labelpad=10)
            ax.set_ylabel('True Label', labelpad=10)
            ax.set_title(title, fontweight='bold')

            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    val = f'{data[i,j]:{fmt}}'
                    color = 'white' if data[i,j] > thresh else 'black'
                    ax.text(j, i, val, ha='center', va='center',
                            color=color, fontsize=max(6, 10 - n // 3))

        fig.suptitle('Confusion Matrix – Test Set',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '05_confusion_matrix')

    # ════════════════════════════════════════════════════════════════
    # 6. PER-CLASS F1 BAR CHART
    # ════════════════════════════════════════════════════════════════
    def plot_f1_per_class(self, y_true, y_pred):
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))),
            zero_division=0)

        n   = len(self.labels_list)
        x   = np.arange(n)
        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))

        bars = ax.bar(x, f1 * 100, color=C_TRAIN, alpha=0.85,
                      edgecolor='white', linewidth=0.8, zorder=3)

        # Thêm giá trị trên cột
        for bar, v in zip(bars, f1):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.0,
                    f'{v*100:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # Đường macro-avg
        macro_f1 = f1.mean() * 100
        ax.axhline(macro_f1, color=C_VAL, lw=2, ls='--',
                   label=f'Macro-avg F1: {macro_f1:.1f}%')

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels_list, rotation=45,
                           ha='right', fontsize=9)
        ax.set_xlabel('Class Label')
        ax.set_ylabel('F1-Score (%)')
        ax.set_title('Per-Class F1-Score', fontweight='bold')
        ax.set_ylim(0, 115)
        ax.legend(framealpha=0.9)

        # Số mẫu dưới x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'n={s}' for s in sup],
                             rotation=45, ha='left', fontsize=7,
                             color='gray')
        ax2.spines['top'].set_visible(False)

        fig.tight_layout()
        return self._save(fig, '06_f1_per_class')

    # ════════════════════════════════════════════════════════════════
    # 7. PRECISION / RECALL / F1 GROUPED BAR
    # ════════════════════════════════════════════════════════════════
    def plot_precision_recall_f1(self, y_true, y_pred):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))),
            zero_division=0)

        n   = len(self.labels_list)
        x   = np.arange(n)
        w   = 0.28
        fig, ax = plt.subplots(figsize=(max(10, n * 0.95), 6))

        ax.bar(x - w, prec * 100, w, label='Precision',
               color='#5E81AC', alpha=0.85, edgecolor='white')
        ax.bar(x,     rec  * 100, w, label='Recall',
               color='#A3BE8C', alpha=0.85, edgecolor='white')
        ax.bar(x + w, f1   * 100, w, label='F1-Score',
               color='#BF616A', alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels_list, rotation=45,
                           ha='right', fontsize=9)
        ax.set_xlabel('Class Label')
        ax.set_ylabel('Score (%)')
        ax.set_title('Precision, Recall & F1-Score per Class',
                     fontweight='bold')
        ax.set_ylim(0, 115)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '07_precision_recall_f1')

    # ════════════════════════════════════════════════════════════════
    # 8. ROC CURVE (One-vs-Rest)
    # ════════════════════════════════════════════════════════════════
    def plot_roc(self, y_true, y_proba):
        n_cls = len(self.labels_list)
        y_bin = label_binarize(y_true, classes=list(range(len(self.labels_list))))
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)
            y_bin = np.hstack([1 - y_bin, y_bin])
        fig, ax = plt.subplots(figsize=(8, 7))

        colors = plt.cm.tab20(np.linspace(0, 1, n_cls))
        fpr_all, tpr_all = [], []
    
        for i in range(n_cls):
            if i >= y_bin.shape[1] or y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, color=colors[i], alpha=0.8,
                    label=f'{self.labels_list[i]} (AUC={roc_auc:.2f})')
            fpr_all.append(fpr)
            tpr_all.append(tpr)

        ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve – One-vs-Rest per Class',
                     fontweight='bold')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.05)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '08_roc_curve')

    # ════════════════════════════════════════════════════════════════
    # 9. t-SNE FEATURE EMBEDDING
    # ════════════════════════════════════════════════════════════════
    def plot_tsne(self, embeddings, y_true, perplexity=30):
        """
        embeddings: numpy (N, D) — feature vectors trước lớp classifier
        y_true    : numpy (N,)
        """
        print("  [Chart] Dang tinh t-SNE (co the mat 30-60s) ...")
        n_cls = len(self.labels_list)

        # Giới hạn số mẫu để t-SNE không quá chậm
        MAX_SAMPLES = 2000
        if len(embeddings) > MAX_SAMPLES:
            idx = np.random.choice(len(embeddings), MAX_SAMPLES, replace=False)
            embeddings = embeddings[idx]
            y_true     = y_true[idx]

        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings)//4),
                    random_state=42, max_iter=1000, init='pca')
        emb2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(9, 8))
        colors  = plt.cm.tab20(np.linspace(0, 1, n_cls))

        for i in range(n_cls):
            mask = y_true == i
            if mask.sum() == 0:
                continue
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                       c=[colors[i]], label=self.labels_list[i],
                       s=18, alpha=0.75, edgecolors='none')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of Feature Embeddings',
                     fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, markerscale=2, framealpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        return self._save(fig, '09_tsne_embeddings')

    # ════════════════════════════════════════════════════════════════
    # 10. TEMPORAL ATTENTION HEATMAP
    # ════════════════════════════════════════════════════════════════
    def plot_temporal_attention(self, attn_weights_dict):
        """
        attn_weights_dict: {label_name: np.array (T,)}
        Trọng số attention trung bình theo thời gian cho mỗi class
        """
        n = len(attn_weights_dict)
        if n == 0:
            return None

        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(cols * 5, rows * 3.5))
        axes = np.array(axes).flatten()

        for ax, (label, weights) in zip(axes, attn_weights_dict.items()):
            T = len(weights)
            # Vẽ heatmap 1D
            im = ax.imshow(weights[np.newaxis, :], aspect='auto',
                           cmap='YlOrRd', vmin=0, vmax=weights.max())
            ax.set_yticks([])
            ax.set_xlabel('Frame Index (Time)')
            ax.set_title(f'"{label}"', fontweight='bold', fontsize=10)
            # Overlay line
            ax2 = ax.twinx()
            ax2.plot(range(T), weights, color='#2E86AB',
                     lw=2, alpha=0.85)
            ax2.set_ylim(0, weights.max() * 1.3)
            ax2.set_yticks([])

        # Tắt axes thừa
        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle('Temporal Attention Weights per Sign Class',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '10_temporal_attention')

    # ════════════════════════════════════════════════════════════════
    # 11. ARCHITECTURE DIAGRAM
    # ════════════════════════════════════════════════════════════════
    def plot_architecture(self, cfg):
        fig = plt.figure(figsize=(16, 10))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 16); ax.set_ylim(0, 10)
        ax.axis('off')

        def box(x, y, w, h, label, sub='', color='#5E81AC',
                fontsize=10, alpha=0.9):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle='round,pad=0.1',
                facecolor=color, edgecolor='white',
                linewidth=1.5, alpha=alpha)
            ax.add_patch(rect)
            cy = y + h/2
            if sub:
                ax.text(x + w/2, cy + 0.2, label,
                        ha='center', va='center', fontsize=fontsize,
                        fontweight='bold', color='white')
                ax.text(x + w/2, cy - 0.3, sub,
                        ha='center', va='center', fontsize=7.5,
                        color='#ECEFF4')
            else:
                ax.text(x + w/2, cy, label,
                        ha='center', va='center', fontsize=fontsize,
                        fontweight='bold', color='white')

        def arrow(x1, y, x2, label=''):
            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle='->', color='#4C566A',
                                        lw=1.8))
            if label:
                ax.text((x1+x2)/2, y + 0.15, label, ha='center',
                        fontsize=8, color='#4C566A')

        def varrow(x, y1, y2, label=''):
            ax.annotate('', xy=(x, y2), xytext=(x, y1),
                        arrowprops=dict(arrowstyle='->', color='#4C566A',
                                        lw=1.8))
            if label:
                ax.text(x + 0.15, (y1+y2)/2, label, fontsize=8,
                        color='#4C566A', va='center')

        # ── Input ──
        box(0.3, 4.5, 2.2, 1.0,
            'Input Sequence',
            f'(B, T={cfg.SEQ_LEN}, D={cfg.FEAT_DIM})',
            color='#4C566A')

        arrow(2.5, 5.0, 3.1,
              label=f'Proj → {cfg.D_MODEL}')

        # ── 6 Feature Groups ──
        grp_labels = ['Pose\n(75)', 'Face\n(90)',
                      'L.Hand\n(63)', 'R.Hand\n(63)',
                      'Blend\n(17)', 'Interact\n(19)']
        grp_colors = ['#2E86AB','#3BB273','#E84855',
                      '#F4A261','#A3BE8C','#B48EAD']
        for i, (gl, gc) in enumerate(zip(grp_labels, grp_colors)):
            bx = 3.2
            by = 8.2 - i * 1.45
            box(bx, by, 1.6, 1.1, gl, color=gc, fontsize=8, alpha=0.85)
            # Arrow to spatial transformer
            ax.annotate('', xy=(6.5, 5.4), xytext=(4.8, by + 0.55),
                        arrowprops=dict(arrowstyle='->', color='#4C566A',
                                        lw=0.8, alpha=0.5))

        # ── Spatial Transformer ──
        box(5.0, 3.8, 2.8, 2.4,
            'Spatial Transformer',
            f'{cfg.SPATIAL_LAYERS}L × {cfg.SPATIAL_HEADS}H\n'
            f'Pre-LN | d={cfg.D_MODEL}\n'
            f'Cross-group Attention',
            color='#2E86AB', fontsize=9)
        arrow(7.8, 5.0, 8.4,
              label=f'(B,T,{cfg.D_MODEL})')

        # ── Temporal Transformer ──
        box(8.4, 3.8, 2.8, 2.4,
            'Temporal Transformer',
            f'{cfg.TEMPORAL_LAYERS}L × {cfg.TEMPORAL_HEADS}H\n'
            f'Pre-LN | d={cfg.D_MODEL}\n'
            f'Sinusoidal PE + CLS',
            color='#5E81AC', fontsize=9)
        arrow(11.2, 5.0, 11.8,
              label=f'CLS(B,{cfg.D_MODEL})')

        # ── Avg Pool branch ──
        box(8.4, 1.6, 2.8, 1.2,
            'Temporal Avg Pool',
            f'mean(T) → (B,{cfg.D_MODEL})',
            color='#81A1C1', fontsize=9)

        # Arrows from spatial to avg pool
        ax.annotate('', xy=(9.8, 2.8), xytext=(9.8, 3.8),
                    arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.5))
        arrow(11.2, 2.2, 11.8)

        # ── Fusion ──
        box(11.8, 3.2, 1.8, 1.6,
            'Fusion',
            f'Concat\n(B, {cfg.D_MODEL*2})',
            color='#88C0D0', fontsize=9)
        arrow(13.6, 4.0, 14.2)

        # ── Classifier ──
        box(14.2, 3.4, 1.5, 1.2,
            'MLP',
            f'{cfg.CLASSIFIER_HIDDEN}→{cfg.CLASSIFIER_HIDDEN//2}\n→ num_cls',
            color='#BF616A', fontsize=9)
        varrow(14.95, 4.6, 5.5)
        box(14.2, 5.5, 1.5, 0.9,
            'Output',
            'Logits (B, C)',
            color='#A3BE8C', fontsize=9)

        # Fusion arrows
        ax.annotate('', xy=(12.7, 3.7), xytext=(11.8, 5.0),
                    arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.5))
        ax.annotate('', xy=(12.7, 3.7), xytext=(11.8, 2.2),
                    arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.5))

        # Title
        ax.text(8, 9.6,
                'Dual Transformer Architecture for Vietnamese Sign Language Recognition',
                ha='center', fontsize=14, fontweight='bold', color='#2E3440')
        ax.text(8, 9.1,
                f'Input: (B, T={cfg.SEQ_LEN}, D={cfg.FEAT_DIM})  |  '
                f'd_model={cfg.D_MODEL}  |  Params ≈ '
                f'{_count_params(cfg):,}',
                ha='center', fontsize=10, color='#4C566A')

        return self._save(fig, '11_architecture_diagram', dpi=200)

    # ════════════════════════════════════════════════════════════════
    # 12. DATASET DISTRIBUTION
    # ════════════════════════════════════════════════════════════════
    def plot_dataset_distribution(self, split_counts: dict):
        """
        split_counts: {'ClassName': {'train': n, 'val': n, 'test': n}}
        """
        labels = list(split_counts.keys())
        n      = len(labels)
        train_c = [split_counts[l].get('train', 0) for l in labels]
        val_c   = [split_counts[l].get('val',   0) for l in labels]
        test_c  = [split_counts[l].get('test',  0) for l in labels]

        x  = np.arange(n)
        w  = 0.28
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n), 6))

        # Grouped bar
        ax1.bar(x - w, train_c, w, label='Train',
                color=C_TRAIN, alpha=0.85)
        ax1.bar(x,     val_c,   w, label='Val',
                color=C_VAL,   alpha=0.85)
        ax1.bar(x + w, test_c,  w, label='Test',
                color=C_TEST,  alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('(a) Sample Count per Split', fontweight='bold')
        ax1.legend(framealpha=0.9)

        # Stacked bar (%)
        total = np.array(train_c) + np.array(val_c) + np.array(test_c)
        total = np.where(total == 0, 1, total)   # avoid div/0
        ax2.bar(x, np.array(train_c)/total*100,
                color=C_TRAIN, alpha=0.85, label='Train')
        ax2.bar(x, np.array(val_c)  /total*100,
                bottom=np.array(train_c)/total*100,
                color=C_VAL,   alpha=0.85, label='Val')
        ax2.bar(x, np.array(test_c) /total*100,
                bottom=(np.array(train_c)+np.array(val_c))/total*100,
                color=C_TEST,  alpha=0.85, label='Test')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Proportion (%)')
        ax2.set_title('(b) Split Proportion per Class', fontweight='bold')
        ax2.set_ylim(0, 108)
        ax2.legend(framealpha=0.9)

        fig.suptitle('Dataset Distribution', fontsize=13, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '12_dataset_distribution')

    # ════════════════════════════════════════════════════════════════
    # 13. TRAINING SUMMARY (1 trang)
    # ════════════════════════════════════════════════════════════════
    def plot_summary(self, test_acc, y_true, y_pred, cfg):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(len(self.labels_list))),
            average='macro', zero_division=0)

        fig = plt.figure(figsize=(18, 11))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                wspace=0.35, hspace=0.4)

        # ── (a) Loss ──
        ax1 = fig.add_subplot(gs[0, 0])
        ep  = self.history['epoch']
        ax1.plot(ep, self.history['train_loss'],
                 color=C_TRAIN, lw=1.8, label='Train')
        ax1.plot(ep, self.history['val_loss'],
                 color=C_VAL, lw=1.8, label='Val')
        ax1.set_title('(a) Loss', fontweight='bold')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(fontsize=8, framealpha=0.9)

        # ── (b) Accuracy ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(ep, self.history['train_acc'],
                 color=C_TRAIN, lw=1.8, label='Train')
        ax2.plot(ep, self.history['val_acc'],
                 color=C_VAL, lw=1.8, label='Val')
        ax2.set_title('(b) Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)')
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=8, framealpha=0.9)

        # ── (c) Confusion Matrix (normalized) ──
        ax3 = fig.add_subplot(gs[0, 2])
        cm  = confusion_matrix(y_true, y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            cmn = np.where(cm.sum(1, keepdims=True) == 0, 0,
                           cm / cm.sum(1, keepdims=True) * 100)
        im = ax3.imshow(cmn, cmap=CMAP_CM, vmin=0, vmax=100)
        ax3.set_title('(c) Confusion Matrix (%)', fontweight='bold')
        ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')
        tick_l = [l[:6] for l in self.labels_list]
        ax3.set_xticks(range(len(tick_l)))
        ax3.set_yticks(range(len(tick_l)))
        ax3.set_xticklabels(tick_l, rotation=45, ha='right', fontsize=7)
        ax3.set_yticklabels(tick_l, fontsize=7)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # ── (d) F1 per class ──
        ax4 = fig.add_subplot(gs[1, 0:2])
        _, _, f1_pc, sup = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(len(self.labels_list))),
            zero_division=0)
        x = np.arange(len(self.labels_list))
        ax4.bar(x, f1_pc * 100, color=C_TRAIN, alpha=0.85,
                edgecolor='white')
        ax4.axhline(f1 * 100, color=C_VAL, lw=2, ls='--',
                    label=f'Macro F1: {f1*100:.1f}%')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.labels_list, rotation=45,
                            ha='right', fontsize=8)
        ax4.set_ylim(0, 115)
        ax4.set_ylabel('F1-Score (%)')
        ax4.set_title('(d) Per-Class F1-Score', fontweight='bold')
        ax4.legend(fontsize=9, framealpha=0.9)

        # ── (e) Metrics text box ──
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        best_val = max(self.history['val_acc'])
        total_ep = ep[-1]

        summary_text = (
            f"Model: Dual Transformer\n"
            f"{'─'*30}\n"
            f"Seq Len    : {cfg.SEQ_LEN} frames\n"
            f"Feat Dim   : {cfg.FEAT_DIM}\n"
            f"d_model    : {cfg.D_MODEL}\n"
            f"Spatial    : {cfg.SPATIAL_LAYERS}L × {cfg.SPATIAL_HEADS}H\n"
            f"Temporal   : {cfg.TEMPORAL_LAYERS}L × {cfg.TEMPORAL_HEADS}H\n"
            f"Params     : {_count_params(cfg):,}\n"
            f"{'─'*30}\n"
            f"Epochs     : {total_ep}\n"
            f"Best Val   : {best_val:.1f}%\n"
            f"Test Acc   : {test_acc*100:.2f}%\n"
            f"Macro Prec : {prec*100:.1f}%\n"
            f"Macro Rec  : {rec*100:.1f}%\n"
            f"Macro F1   : {f1*100:.1f}%\n"
            f"Classes    : {len(self.labels_list)}"
        )
        ax5.text(0.05, 0.97, summary_text,
                 transform=ax5.transAxes,
                 va='top', ha='left', fontsize=10,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#ECEFF4',
                           edgecolor='#4C566A', linewidth=1.5))
        ax5.set_title('(e) Summary', fontweight='bold')

        fig.suptitle(
            'Dual Transformer – Full Training & Evaluation Summary\n'
            'Vietnamese Sign Language Recognition',
            fontsize=13, fontweight='bold', y=1.01)

        return self._save(fig, '13_training_summary')

    # ════════════════════════════════════════════════════════════════
    # MAIN ENTRY: gọi sau khi train xong
    # ════════════════════════════════════════════════════════════════
    def plot_all(self, model, test_loader, device, cfg,
                 split_counts=None):
        """
        Xuất toàn bộ biểu đồ.

        model        : DualTransformer đã load best checkpoint
        test_loader  : DataLoader test set
        device       : 'cuda' / 'cpu'
        cfg          : Config object
        split_counts : dict cho plot_dataset_distribution (optional)
        """
        print("\n" + "="*60)
        print(" XUAT BIEU DO BAO CAO ".center(60))
        print("="*60)

        # ── Thu thập predictions, probabilities, embeddings ──
        model.eval()
        all_labels, all_preds, all_proba, all_embeds = [], [], [], []

        # Hook để lấy embedding trước classifier
        _embed_buf = []
        def _hook(module, inp, out):
            _embed_buf.append(inp[0].detach().cpu())
        handle = model.classifier[0].register_forward_hook(_hook)

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                logits = model(X)
                proba  = F.softmax(logits, dim=-1).cpu().numpy()
                preds  = logits.argmax(-1).cpu().numpy()
                all_labels.extend(y.numpy())
                all_preds.extend(preds)
                all_proba.extend(proba)

        handle.remove()
        all_embeds = torch.cat(_embed_buf, dim=0).numpy() if _embed_buf else None
        y_true  = np.array(all_labels)
        y_pred  = np.array(all_preds)
        y_proba = np.array(all_proba)
        test_acc = (y_true == y_pred).mean()

        # ── Xuất biểu đồ ──
        paths = {}
        paths['loss']      = self.plot_loss()
        paths['accuracy']  = self.plot_accuracy()
        paths['lr']        = self.plot_lr()
        paths['combo']     = self.plot_training_combo()
        paths['cm']        = self.plot_confusion_matrix(y_true, y_pred)
        paths['f1']        = self.plot_f1_per_class(y_true, y_pred)
        paths['prf']       = self.plot_precision_recall_f1(y_true, y_pred)
        paths['roc']       = self.plot_roc(y_true, y_proba)

        if all_embeds is not None:
            paths['tsne'] = self.plot_tsne(all_embeds, y_true)

        paths['arch']      = self.plot_architecture(cfg)
        paths['summary']   = self.plot_summary(test_acc, y_true, y_pred, cfg)

        if split_counts is not None:
            paths['dist'] = self.plot_dataset_distribution(split_counts)

        # ── Index file ──
        index = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_accuracy': float(test_acc),
            'charts': paths,
        }
        idx_path = os.path.join(self.output_dir, 'chart_index.json')
        with open(idx_path, 'w') as f:
            json.dump(index, f, indent=2)

        print(f"\n  Tong so bieu do: {len(paths)}")
        print(f"  Index: {idx_path}")
        print("="*60 + "\n")
        return paths


# ═══════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════

def _count_params(cfg) -> int:
    """Ước tính số params từ cfg (không cần khởi tạo model)"""
    d = cfg.D_MODEL
    # Spatial: 6 proj + transformer
    spatial = (6 * 75 * d + cfg.SPATIAL_LAYERS * (4 * d*d + 2 * d * cfg.SPATIAL_FF_DIM))
    # Temporal: transformer
    temporal = cfg.TEMPORAL_LAYERS * (4 * d*d + 2 * d * cfg.TEMPORAL_FF_DIM)
    # Classifier
    cls = d*2 * cfg.CLASSIFIER_HIDDEN + cfg.CLASSIFIER_HIDDEN * (cfg.CLASSIFIER_HIDDEN//2)
    return spatial + temporal + cls


def load_history_from_json(log_path: str) -> dict:
    """Load lịch sử training từ file log JSON để vẽ lại biểu đồ"""
    with open(log_path, 'r') as f:
        return json.load(f)


# Import needed inside Visualizer
from datetime import datetime