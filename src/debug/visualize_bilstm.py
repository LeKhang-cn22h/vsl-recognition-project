"""
visualize_bilstm.py - Training Visualization cho BiLSTM
=========================================================
Tái sử dụng logic từ visualize_training.py (Dual Transformer) nhưng:
  - Tên file: bilstm_01_loss_curve.png, bilstm_02_... (không đụng file DualTransformer)
  - Title/label ghi "BiLSTM"
  - plot_architecture() vẽ sơ đồ BiLSTM + Attention
  - plot_all() hook đúng vào BiLSTMClassifier

Cách dùng trong train_bilstm.py:
    from visualize_bilstm import BiLSTMVisualizer
    viz = BiLSTMVisualizer(label_map, output_dir='charts')
    viz.update(epoch, train_loss, val_loss, train_acc, val_acc, lr)
    viz.plot_all(model, test_loader, device, cfg, split_counts=split_counts)
"""

import os, json, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from datetime import datetime

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

C_TRAIN = '#2E86AB'
C_VAL   = '#E84855'
C_TEST  = '#3BB273'
C_LR    = '#F4A261'
CMAP_CM = LinearSegmentedColormap.from_list(
    'cm_blue', ['#FFFFFF', '#2E86AB', '#1A3A4A'])


class BiLSTMVisualizer:
    PREFIX = 'bilstm_'  # <- khác với visualize_training.py (không có prefix)

    def __init__(self, label_map: dict, output_dir: str = 'charts'):
        self.label_map   = label_map
        self.idx2label   = {v: k for k, v in label_map.items()}
        self.labels_list = [self.idx2label[i] for i in range(len(label_map))]
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = dict(epoch=[], train_loss=[], val_loss=[],
                            train_acc=[], val_acc=[], lr=[])
        print(f"  [BiLSTMVisualizer] Charts -> {output_dir}/bilstm_*")

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc * 100)
        self.history['val_acc'].append(val_acc * 100)
        self.history['lr'].append(lr)

    def _save(self, fig, name, dpi=300):
        path = os.path.join(self.output_dir, f'{self.PREFIX}{name}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [Chart] {path}")
        return path

    # 1. Loss
    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = self.history['epoch']
        ax.plot(ep, self.history['train_loss'], color=C_TRAIN, lw=2,
                label='Train Loss', marker='o', markersize=3,
                markevery=max(1, len(ep)//15))
        ax.plot(ep, self.history['val_loss'], color=C_VAL, lw=2,
                label='Val Loss', marker='s', markersize=3,
                markevery=max(1, len(ep)//15))
        best_ep = ep[int(np.argmin(self.history['val_loss']))]
        best_vl = min(self.history['val_loss'])
        ax.axvline(best_ep, color=C_VAL, lw=1, ls=':', alpha=0.7)
        ax.annotate(f'Best\nEpoch {best_ep}', xy=(best_ep, best_vl),
                    xytext=(best_ep + max(1, len(ep)*0.05), best_vl),
                    fontsize=9, color=C_VAL,
                    arrowprops=dict(arrowstyle='->', color=C_VAL, lw=1.2))
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Cross-Entropy)')
        ax.set_title('BiLSTM – Training & Validation Loss', fontweight='bold')
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '01_loss_curve')

    # 2. Accuracy
    def plot_accuracy(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = self.history['epoch']
        ax.plot(ep, self.history['train_acc'], color=C_TRAIN, lw=2,
                label='Train Accuracy', marker='o', markersize=3,
                markevery=max(1, len(ep)//15))
        ax.plot(ep, self.history['val_acc'], color=C_VAL, lw=2,
                label='Val Accuracy', marker='s', markersize=3,
                markevery=max(1, len(ep)//15))
        best_ep  = ep[int(np.argmax(self.history['val_acc']))]
        best_acc = max(self.history['val_acc'])
        ax.axvline(best_ep, color=C_VAL, lw=1, ls=':', alpha=0.7)
        ax.annotate(f'{best_acc:.1f}%\nEpoch {best_ep}',
                    xy=(best_ep, best_acc),
                    xytext=(best_ep + max(1, len(ep)*0.05), best_acc - 5),
                    fontsize=9, color=C_VAL,
                    arrowprops=dict(arrowstyle='->', color=C_VAL, lw=1.2))
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
        ax.set_title('BiLSTM – Training & Validation Accuracy', fontweight='bold')
        ax.set_ylim(0, 105); ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '02_accuracy_curve')

    # 3. LR
    def plot_lr(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        ep = self.history['epoch']
        ax.plot(ep, self.history['lr'], color=C_LR, lw=2, label='Learning Rate')
        ax.fill_between(ep, self.history['lr'], alpha=0.15, color=C_LR)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.set_title('BiLSTM – Learning Rate Schedule', fontweight='bold')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '03_lr_schedule')

    # 4. Combo
    def plot_training_combo(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ep = self.history['epoch']
        for ax, key, ylabel, title in [
            (ax1, 'loss', 'Loss', '(a) Loss Curve'),
            (ax2, 'acc',  'Accuracy (%)', '(b) Accuracy Curve'),
        ]:
            ax.plot(ep, self.history[f'train_{key}'], color=C_TRAIN, lw=2,
                    label='Train', marker='o', markersize=2,
                    markevery=max(1, len(ep)//15))
            ax.plot(ep, self.history[f'val_{key}'], color=C_VAL, lw=2,
                    label='Val', marker='s', markersize=2,
                    markevery=max(1, len(ep)//15))
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.legend(framealpha=0.9)
            if key == 'acc':
                ax.set_ylim(0, 105)
        fig.suptitle('BiLSTM – Training History',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self._save(fig, '04_training_combo')

    # 5. Confusion Matrix
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        n  = len(self.labels_list)
        tick_labels = [l.replace('_', '\n') for l in self.labels_list]
        fig_size = max(8, n * 0.85)
        fig, axes = plt.subplots(1, 2, figsize=(fig_size*2+1, fig_size))
        for ax, norm, title in zip(axes,
                [True, False], ['(a) Normalized (%)', '(b) Raw Count']):
            if norm:
                with np.errstate(divide='ignore', invalid='ignore'):
                    data = np.where(cm.sum(axis=1,keepdims=True)==0, 0,
                                    cm/cm.sum(axis=1,keepdims=True)*100)
                fmt = '.1f'; vmax = 100
            else:
                data = cm; fmt = 'd'; vmax = cm.max()
            im = ax.imshow(data, interpolation='nearest',
                           cmap=CMAP_CM, vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlabel('Predicted Label', labelpad=10)
            ax.set_ylabel('True Label', labelpad=10)
            ax.set_title(title, fontweight='bold')
            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    color = 'white' if data[i,j] > thresh else 'black'
                    ax.text(j, i, f'{data[i,j]:{fmt}}',
                            ha='center', va='center', color=color,
                            fontsize=max(6, 10-n//3))
        fig.suptitle('BiLSTM – Confusion Matrix (Test Set)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '05_confusion_matrix')

    # 6. F1 per class
    def plot_f1_per_class(self, y_true, y_pred):
        _, _, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))),
            zero_division=0)
        n = len(self.labels_list); x = np.arange(n)
        fig, ax = plt.subplots(figsize=(max(10, n*0.9), 6))
        bars = ax.bar(x, f1*100, color=C_TRAIN, alpha=0.85,
                      edgecolor='white', zorder=3)
        for bar, v in zip(bars, f1):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.0,
                    f'{v*100:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        macro_f1 = f1.mean()*100
        ax.axhline(macro_f1, color=C_VAL, lw=2, ls='--',
                   label=f'Macro-avg F1: {macro_f1:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels_list, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Class Label'); ax.set_ylabel('F1-Score (%)')
        ax.set_title('BiLSTM – Per-Class F1-Score', fontweight='bold')
        ax.set_ylim(0, 115); ax.legend(framealpha=0.9)
        ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'n={s}' for s in sup],
                             rotation=45, ha='left', fontsize=7, color='gray')
        ax2.spines['top'].set_visible(False)
        fig.tight_layout()
        return self._save(fig, '06_f1_per_class')

    # 7. Precision/Recall/F1
    def plot_precision_recall_f1(self, y_true, y_pred):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))),
            zero_division=0)
        n = len(self.labels_list); x = np.arange(n); w = 0.28
        fig, ax = plt.subplots(figsize=(max(10, n*0.95), 6))
        ax.bar(x-w, prec*100, w, label='Precision',
               color='#5E81AC', alpha=0.85, edgecolor='white')
        ax.bar(x,   rec*100,  w, label='Recall',
               color='#A3BE8C', alpha=0.85, edgecolor='white')
        ax.bar(x+w, f1*100,   w, label='F1-Score',
               color='#BF616A', alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels_list, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Class Label'); ax.set_ylabel('Score (%)')
        ax.set_title('BiLSTM – Precision, Recall & F1-Score per Class',
                     fontweight='bold')
        ax.set_ylim(0, 115); ax.legend(framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '07_precision_recall_f1')

    # 8. ROC
    def plot_roc(self, y_true, y_proba):
        n_cls = len(self.labels_list)
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        if y_bin.ndim == 1:
            y_bin = np.hstack([1-y_bin.reshape(-1,1), y_bin.reshape(-1,1)])
        fig, ax = plt.subplots(figsize=(8, 7))
        colors  = plt.cm.tab20(np.linspace(0, 1, n_cls))
        for i in range(n_cls):
            if i >= y_bin.shape[1] or y_bin[:,i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_proba[:,i])
            ax.plot(fpr, tpr, lw=1.5, color=colors[i], alpha=0.8,
                    label=f'{self.labels_list[i]} (AUC={auc(fpr,tpr):.2f})')
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random (AUC=0.50)')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('BiLSTM – ROC Curve (One-vs-Rest)', fontweight='bold')
        ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.05)
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left',
                  fontsize=8, framealpha=0.9)
        fig.tight_layout()
        return self._save(fig, '08_roc_curve')

    # 9. t-SNE
    def plot_tsne(self, embeddings, y_true, perplexity=30):
        print("  [Chart] Dang tinh t-SNE ...")
        n_cls = len(self.labels_list)
        MAX_SAMPLES = 2000
        if len(embeddings) > MAX_SAMPLES:
            idx = np.random.choice(len(embeddings), MAX_SAMPLES, replace=False)
            embeddings = embeddings[idx]; y_true = y_true[idx]
        tsne  = TSNE(n_components=2,
                     perplexity=min(perplexity, len(embeddings)//4),
                     random_state=42, max_iter=1000, init='pca')
        emb2d = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(9, 8))
        colors  = plt.cm.tab20(np.linspace(0, 1, n_cls))
        for i in range(n_cls):
            mask = y_true == i
            if mask.sum() == 0: continue
            ax.scatter(emb2d[mask,0], emb2d[mask,1], c=[colors[i]],
                       label=self.labels_list[i], s=18, alpha=0.75,
                       edgecolors='none')
        ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
        ax.set_title('BiLSTM – t-SNE Feature Embeddings', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left',
                  fontsize=8, markerscale=2, framealpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        return self._save(fig, '09_tsne_embeddings')

    # 10. Temporal Attention
    def plot_temporal_attention(self, attn_weights_dict):
        n = len(attn_weights_dict)
        if n == 0: return None
        cols = min(3, n); rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3.5))
        axes = np.array(axes).flatten()
        for ax, (label, weights) in zip(axes, attn_weights_dict.items()):
            T  = len(weights)
            ax.imshow(weights[np.newaxis,:], aspect='auto',
                      cmap='YlOrRd', vmin=0, vmax=weights.max())
            ax.set_yticks([]); ax.set_xlabel('Frame Index (Time)')
            ax.set_title(f'"{label}"', fontweight='bold', fontsize=10)
            ax2 = ax.twinx()
            ax2.plot(range(T), weights, color='#2E86AB', lw=2, alpha=0.85)
            ax2.set_ylim(0, weights.max()*1.3); ax2.set_yticks([])
        for ax in axes[n:]: ax.set_visible(False)
        fig.suptitle('BiLSTM – Temporal Attention per Sign Class',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '10_temporal_attention')

    # 11. Architecture Diagram (BiLSTM)
    def plot_architecture(self, cfg):
        fig = plt.figure(figsize=(16, 9))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')

        def box(x, y, w, h, label, sub='', color='#5E81AC',
                fontsize=10, alpha=0.9):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x,y), w, h, boxstyle='round,pad=0.1',
                facecolor=color, edgecolor='white',
                linewidth=1.5, alpha=alpha))
            cy = y + h/2
            ax.text(x+w/2, cy+(0.2 if sub else 0), label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color='white')
            if sub:
                ax.text(x+w/2, cy-0.3, sub, ha='center', va='center',
                        fontsize=7.5, color='#ECEFF4')

        def arrow(x1, y, x2, label=''):
            ax.annotate('', xy=(x2,y), xytext=(x1,y),
                        arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.8))
            if label:
                ax.text((x1+x2)/2, y+0.15, label,
                        ha='center', fontsize=8, color='#4C566A')

        def varrow(x, y1, y2):
            ax.annotate('', xy=(x,y2), xytext=(x,y1),
                        arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.8))

        box(0.3, 3.8, 2.0, 1.0, 'Input',
            f'(B,T={cfg.SEQ_LEN},D={cfg.FEAT_DIM})', color='#4C566A')
        arrow(2.3, 4.3, 3.0)

        box(3.0, 3.4, 2.2, 1.6, 'Input Proj',
            f'Linear→{cfg.HIDDEN_DIM}\nLayerNorm|ReLU', color='#81A1C1')
        arrow(5.2, 4.2, 5.9, f'(B,T,{cfg.HIDDEN_DIM})')

        colors_lstm = ['#2E86AB','#3A7EAB','#4676AB']
        for i in range(min(cfg.NUM_LAYERS, 3)):
            by = 5.6 - i * 1.3
            box(5.9, by, 2.8, 1.1,
                f'BiLSTM Layer {i+1}',
                f'hidden={cfg.HIDDEN_DIM}x2', color=colors_lstm[i])
            if i < min(cfg.NUM_LAYERS,3)-1:
                varrow(7.3, by, by-0.2)
        if cfg.NUM_LAYERS > 3:
            ax.text(7.3, 2.2, f'... ({cfg.NUM_LAYERS} layers)',
                    ha='center', fontsize=8, color='#4C566A', style='italic')

        arrow(8.7, 4.1, 9.4, f'(B,T,{cfg.HIDDEN_DIM*2})')

        if cfg.USE_ATTENTION:
            box(9.4, 3.3, 2.5, 1.6, 'Attention',
                f'Linear({cfg.HIDDEN_DIM*2},1)\nsoftmax→context',
                color='#F4A261')
            arrow(11.9, 4.1, 12.5, f'ctx(B,{cfg.HIDDEN_DIM*2})')
            box(9.4, 1.5, 2.5, 1.0, 'Last Hidden',
                f'hn→(B,{cfg.HIDDEN_DIM*2})', color='#5E81AC')
            ax.annotate('', xy=(12.5,3.0), xytext=(11.9,2.0),
                        arrowprops=dict(arrowstyle='->', color='#4C566A', lw=1.2))
            box(12.5, 3.0, 1.8, 1.4, 'Concat',
                f'(B,{cfg.HIDDEN_DIM*4})', color='#88C0D0')
            arrow(14.3, 3.7, 14.8)
        else:
            box(9.4, 3.3, 2.5, 1.2, 'Last Hidden',
                f'hn→(B,{cfg.HIDDEN_DIM*2})', color='#5E81AC')
            arrow(11.9, 3.9, 12.5)
            box(12.5, 3.0, 1.8, 1.4, 'Feature',
                f'(B,{cfg.HIDDEN_DIM*2})', color='#88C0D0')
            arrow(14.3, 3.7, 14.8)

        mid = max(len(self.labels_list)*4, 128)
        box(14.8, 3.1, 1.0, 1.2, 'MLP',
            f'{mid}→{len(self.labels_list)}', color='#BF616A', fontsize=9)
        varrow(15.3, 4.3, 5.2)
        box(14.8, 5.2, 1.0, 0.9, 'Output',
            f'(B,{len(self.labels_list)})', color='#A3BE8C', fontsize=9)

        dirs = 2 if cfg.BIDIRECTIONAL else 1
        ax.text(8, 8.5,
                'Bidirectional LSTM – Vietnamese Sign Language Recognition',
                ha='center', fontsize=13, fontweight='bold', color='#2E3440')
        ax.text(8, 7.9,
                f'Input:(B,T={cfg.SEQ_LEN},D={cfg.FEAT_DIM})  '
                f'Hidden={cfg.HIDDEN_DIM}x{dirs}  '
                f'Layers={cfg.NUM_LAYERS}  Attention={cfg.USE_ATTENTION}',
                ha='center', fontsize=10, color='#4C566A')

        return self._save(fig, '11_architecture_diagram', dpi=200)

    # 12. Dataset distribution
    def plot_dataset_distribution(self, split_counts: dict):
        labels  = list(split_counts.keys()); n = len(labels)
        train_c = [split_counts[l].get('train',0) for l in labels]
        val_c   = [split_counts[l].get('val',  0) for l in labels]
        test_c  = [split_counts[l].get('test', 0) for l in labels]
        x = np.arange(n); w = 0.28
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12,n), 6))
        ax1.bar(x-w, train_c, w, label='Train', color=C_TRAIN, alpha=0.85)
        ax1.bar(x,   val_c,   w, label='Val',   color=C_VAL,   alpha=0.85)
        ax1.bar(x+w, test_c,  w, label='Test',  color=C_TEST,  alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Samples'); ax1.set_title('(a) Count per Split', fontweight='bold')
        ax1.legend(framealpha=0.9)
        total = np.array(train_c)+np.array(val_c)+np.array(test_c)
        total = np.where(total==0,1,total)
        ax2.bar(x,   np.array(train_c)/total*100, color=C_TRAIN, alpha=0.85, label='Train')
        ax2.bar(x,   np.array(val_c)  /total*100,
                bottom=np.array(train_c)/total*100, color=C_VAL, alpha=0.85, label='Val')
        ax2.bar(x,   np.array(test_c) /total*100,
                bottom=(np.array(train_c)+np.array(val_c))/total*100,
                color=C_TEST, alpha=0.85, label='Test')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Proportion (%)'); ax2.set_ylim(0,108)
        ax2.set_title('(b) Split Proportion', fontweight='bold')
        ax2.legend(framealpha=0.9)
        fig.suptitle('BiLSTM – Dataset Distribution', fontsize=13, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '12_dataset_distribution')

    # 13. Summary
    def plot_summary(self, test_acc, y_true, y_pred, cfg):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))),
            average='macro', zero_division=0)
        fig = plt.figure(figsize=(18, 11))
        gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)
        ep  = self.history['epoch']

        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(ep, self.history['train_loss'], color=C_TRAIN, lw=1.8, label='Train')
        ax1.plot(ep, self.history['val_loss'],   color=C_VAL,   lw=1.8, label='Val')
        ax1.set_title('(a) Loss', fontweight='bold')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0,1])
        ax2.plot(ep, self.history['train_acc'], color=C_TRAIN, lw=1.8, label='Train')
        ax2.plot(ep, self.history['val_acc'],   color=C_VAL,   lw=1.8, label='Val')
        ax2.set_title('(b) Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)'); ax2.set_ylim(0,105)
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[0,2])
        cm  = confusion_matrix(y_true, y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            cmn = np.where(cm.sum(1,keepdims=True)==0, 0,
                           cm/cm.sum(1,keepdims=True)*100)
        im = ax3.imshow(cmn, cmap=CMAP_CM, vmin=0, vmax=100)
        ax3.set_title('(c) Confusion Matrix (%)', fontweight='bold')
        tick_l = [l[:6] for l in self.labels_list]
        ax3.set_xticks(range(len(tick_l))); ax3.set_yticks(range(len(tick_l)))
        ax3.set_xticklabels(tick_l, rotation=45, ha='right', fontsize=7)
        ax3.set_yticklabels(tick_l, fontsize=7)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs[1,0:2])
        _, _, f1_pc, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.labels_list))), zero_division=0)
        x = np.arange(len(self.labels_list))
        ax4.bar(x, f1_pc*100, color=C_TRAIN, alpha=0.85, edgecolor='white')
        ax4.axhline(f1*100, color=C_VAL, lw=2, ls='--',
                    label=f'Macro F1: {f1*100:.1f}%')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.labels_list, rotation=45, ha='right', fontsize=8)
        ax4.set_ylim(0,115); ax4.set_ylabel('F1-Score (%)')
        ax4.set_title('(d) Per-Class F1-Score', fontweight='bold')
        ax4.legend(fontsize=9)

        ax5 = fig.add_subplot(gs[1,2]); ax5.axis('off')
        dirs = 2 if cfg.BIDIRECTIONAL else 1
        txt = (
            f"Model: Bidirectional LSTM\n"
            f"{'─'*30}\n"
            f"Seq Len   : {cfg.SEQ_LEN} frames\n"
            f"Feat Dim  : {cfg.FEAT_DIM}\n"
            f"Hidden    : {cfg.HIDDEN_DIM} x {dirs} dirs\n"
            f"Layers    : {cfg.NUM_LAYERS}\n"
            f"Attention : {cfg.USE_ATTENTION}\n"
            f"Scheduler : {cfg.LR_SCHEDULER.upper()}\n"
            f"{'─'*30}\n"
            f"Epochs    : {ep[-1]}\n"
            f"Best Val  : {max(self.history['val_acc']):.1f}%\n"
            f"Test Acc  : {test_acc*100:.2f}%\n"
            f"Macro P   : {prec*100:.1f}%\n"
            f"Macro R   : {rec*100:.1f}%\n"
            f"Macro F1  : {f1*100:.1f}%\n"
            f"Classes   : {len(self.labels_list)}"
        )
        ax5.text(0.05, 0.97, txt, transform=ax5.transAxes,
                 va='top', ha='left', fontsize=10, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF4',
                           edgecolor='#4C566A', linewidth=1.5))
        ax5.set_title('(e) Summary', fontweight='bold')
        fig.suptitle('BiLSTM – Full Training & Evaluation Summary\n'
                     'Vietnamese Sign Language Recognition',
                     fontsize=13, fontweight='bold', y=1.01)
        return self._save(fig, '13_training_summary')

    # MAIN
    def plot_all(self, model, test_loader, device, cfg, split_counts=None):
        print("\n" + "="*60)
        print(" XUAT BIEU DO BILSTM ".center(60))
        print("="*60)

        model.eval()
        all_labels, all_preds, all_proba = [], [], []
        attn_by_class = {}
        _embed_buf = []

        def _hook(module, inp, out):
            _embed_buf.append(inp[0].detach().cpu())

        # Hook vào Linear đầu tiên trong classifier Sequential của BiLSTM
        handle = model.classifier[1].register_forward_hook(_hook)

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                if cfg.USE_ATTENTION:
                    logits, attn = model(X, return_attention=True)
                    preds = logits.argmax(-1).cpu().numpy()
                    for pred_i, attn_i in zip(preds, attn.cpu().numpy()):
                        lname = self.idx2label.get(int(pred_i), str(pred_i))
                        attn_by_class.setdefault(lname, []).append(attn_i)
                else:
                    logits = model(X)
                    preds  = logits.argmax(-1).cpu().numpy()
                proba = F.softmax(logits, dim=-1).cpu().numpy()
                all_labels.extend(y.numpy())
                all_preds.extend(preds)
                all_proba.extend(proba)

        handle.remove()

        y_true   = np.array(all_labels)
        y_pred   = np.array(all_preds)
        y_proba  = np.array(all_proba)
        test_acc = (y_true == y_pred).mean()
        all_embeds = torch.cat(_embed_buf, dim=0).numpy() if _embed_buf else None
        attn_mean  = {k: np.mean(v, axis=0)
                      for k, v in attn_by_class.items() if len(v) > 0}

        paths = {}
        paths['loss']     = self.plot_loss()
        paths['accuracy'] = self.plot_accuracy()
        paths['lr']       = self.plot_lr()
        paths['combo']    = self.plot_training_combo()
        paths['cm']       = self.plot_confusion_matrix(y_true, y_pred)
        paths['f1']       = self.plot_f1_per_class(y_true, y_pred)
        paths['prf']      = self.plot_precision_recall_f1(y_true, y_pred)
        paths['roc']      = self.plot_roc(y_true, y_proba)
        if all_embeds is not None:
            paths['tsne'] = self.plot_tsne(all_embeds, y_true)
        if attn_mean:
            paths['attn'] = self.plot_temporal_attention(attn_mean)
        paths['arch']    = self.plot_architecture(cfg)
        paths['summary'] = self.plot_summary(test_acc, y_true, y_pred, cfg)
        if split_counts is not None:
            paths['dist'] = self.plot_dataset_distribution(split_counts)

        idx_path = os.path.join(self.output_dir, 'bilstm_chart_index.json')
        with open(idx_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': 'BiLSTM',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_accuracy': float(test_acc),
                'charts': paths,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n  Tong so bieu do: {len(paths)}")
        print(f"  Index: {idx_path}")
        print("="*60 + "\n")
        return paths