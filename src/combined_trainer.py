"""
VSL Combined Trainer
Huấn luyện mô hình nhận diện kết hợp: biểu cảm mặt + hành động tay → ký hiệu
Input shape mỗi sample: (30, 135)
  - 126: hand landmarks (2 tay × 21 × 3)
  -   9: face features  (EAR, MAR, BROW, ...)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
import json
import time
import matplotlib
matplotlib.use('Agg')   # không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data', 'combined_raw')
MODEL_DIR   = os.path.join(BASE_DIR, '..', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'combined')

SEQUENCE_LENGTH  = 30
N_HAND_FEATURES  = 126
N_FACE_FEATURES  = 9
N_TOTAL_FEATURES = N_HAND_FEATURES + N_FACE_FEATURES   # 135


# ══════════════════════════════════════════════════════ Load Data

def load_combined_data(data_dir: str):
    X, y = [], []
    print(f"\n📂 Đang quét: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"  ✗ Không tìm thấy: {data_dir}")
        print("  → Hãy chạy combined_data_collector.py trước!")
        return np.array([]), np.array([])

    folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ])
    if not folders:
        print("  ✗ Không có folder class nào!")
        return np.array([]), np.array([])

    print(f"  Tìm thấy {len(folders)} ký hiệu: {folders}\n")

    stats = {}
    for label in folders:
        label_dir = os.path.join(data_dir, label)
        files = glob.glob(os.path.join(label_dir, '*.npy'))
        ok = 0
        for f in files:
            try:
                seq = np.load(f)
                # Chấp nhận cả shape (30,135) và (30,126) để tương thích ngược
                if seq.shape[0] == SEQUENCE_LENGTH:
                    if seq.shape[1] == N_TOTAL_FEATURES:
                        X.append(seq)
                        y.append(label)
                        ok += 1
                    elif seq.shape[1] == N_HAND_FEATURES:
                        # Pad thêm 9 zeros cho face features (data cũ)
                        pad = np.zeros((SEQUENCE_LENGTH, N_FACE_FEATURES), dtype=np.float32)
                        X.append(np.hstack([seq, pad]))
                        y.append(label)
                        ok += 1
                    else:
                        print(f"  ⚠ Bỏ qua shape lạ {seq.shape}: {os.path.basename(f)}")
                else:
                    print(f"  ⚠ Bỏ qua seq_len khác {seq.shape}: {os.path.basename(f)}")
            except Exception as e:
                print(f"  ✗ Lỗi đọc {f}: {e}")
        stats[label] = ok

    print("  📊 Thống kê:")
    for lbl, cnt in stats.items():
        bar = "█" * max(cnt // 2, 1)
        print(f"    {lbl:<18} {cnt:>4} mẫu  {bar}")
    print(f"\n  Tổng: {len(X)} mẫu, {len(stats)} lớp")

    if len(X) < 20:
        print("\n  ⚠ Quá ít dữ liệu! Hãy thu thập thêm (khuyến nghị >=30 mẫu/lớp).")

    return np.array(X, dtype=np.float32), np.array(y)


# ══════════════════════════════════════════════════════ Model Builders

def _base_input(seq_len, n_feat):
    return keras.Input(shape=(seq_len, n_feat))


@keras.utils.register_keras_serializable(package='VSL')
class FeatureSlice(layers.Layer):
    """Layer slice features theo axis=2 — serialize được, không dùng Lambda."""
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end   = end

    def call(self, x):
        return x[:, :, self.start:self.end]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'start': self.start, 'end': self.end})
        return cfg


def build_dual_stream(seq_len, n_feat, n_classes):
    """
    Dual-stream model: xử lý hand features và face features riêng biệt
    trước khi ghép → phù hợp nhất cho bài toán kết hợp này.
    Dùng FeatureSlice (custom layer đã register) thay Lambda để serialize an toàn.
    """
    inp = keras.Input(shape=(seq_len, n_feat), name='combined_input')

    # Stream 1: Tay (126 features đầu)
    hand_stream = FeatureSlice(0, N_HAND_FEATURES, name='hand_slice')(inp)
    hand_stream = layers.LSTM(128, return_sequences=True, name='hand_lstm1')(hand_stream)
    hand_stream = layers.Dropout(0.3)(hand_stream)
    hand_stream = layers.LSTM(64, return_sequences=False, name='hand_lstm2')(hand_stream)
    hand_stream = layers.Dense(64, activation='relu', name='hand_dense')(hand_stream)

    # Stream 2: Mặt (9 features cuối)
    face_stream = FeatureSlice(N_HAND_FEATURES, N_HAND_FEATURES + N_FACE_FEATURES, name='face_slice')(inp)
    face_stream = layers.LSTM(32, return_sequences=True, name='face_lstm1')(face_stream)
    face_stream = layers.Dropout(0.2)(face_stream)
    face_stream = layers.LSTM(16, return_sequences=False, name='face_lstm2')(face_stream)
    face_stream = layers.Dense(16, activation='relu', name='face_dense')(face_stream)

    # Merge
    merged = layers.Concatenate(name='merge')([hand_stream, face_stream])
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(32, activation='relu')(merged)
    output = layers.Dense(n_classes, activation='softmax')(merged)

    model = keras.Model(inputs=inp, outputs=output, name='DualStream_LSTM')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_attention_combined(seq_len, n_feat, n_classes):
    """Bidirectional LSTM + Multi-head Attention trên toàn bộ 135 features"""
    inp = keras.Input(shape=(seq_len, n_feat))

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs=inp, outputs=out, name='Attention_BiLSTM')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_simple_lstm(seq_len, n_feat, n_classes):
    """Baseline đơn giản để so sánh"""
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_feat)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ], name='Simple_LSTM')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


MODEL_BUILDERS = {
    'dual_stream':        build_dual_stream,        # recommended
    'attention_bilstm':   build_attention_combined,
    'simple_lstm':        build_simple_lstm,
}


# ══════════════════════════════════════════════════════ Train / Eval

def train_model(model, X_tr, y_tr, X_val, y_val):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1),
    ]
    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    return hist, time.time() - t0


def evaluate_model(model, X_te, y_te):
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    return {'loss': loss, 'accuracy': acc, 'y_pred': y_pred}


# ══════════════════════════════════════════════════════ Visualize

def plot_comparison(results, save_path):
    names  = list(results.keys())
    accs   = [results[n]['accuracy'] * 100 for n in names]
    best_i = int(np.argmax(accs))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['gold' if i == best_i else 'steelblue' for i in range(len(names))]
    bars = ax.bar(names, accs, color=colors, edgecolor='navy', linewidth=1.5)
    ax.axhline(np.mean(accs), color='red', ls='--', lw=1.5,
               label=f'Avg {np.mean(accs):.1f}%')
    for i, (b, a) in enumerate(zip(bars, accs)):
        ax.text(b.get_x() + b.get_width()/2, a+0.5,
                f'{a:.1f}%', ha='center', va='bottom',
                fontweight='bold' if i == best_i else 'normal')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Combined Model Comparison (Face + Hand)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_confusion(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)-1)))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues', linewidths=0.5)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_training_curves(histories, names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    for i, (hist, name) in enumerate(zip(histories, names)):
        c = colors[i % len(colors)]
        axes[0].plot(hist.history['val_accuracy'], label=name, color=c, lw=2)
        axes[1].plot(hist.history['val_loss'],     label=name, color=c, lw=2)
    for ax, title, ylabel in zip(
        axes,
        ['Val Accuracy', 'Val Loss'],
        ['Accuracy', 'Loss']
    ):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════ Save

def save_artifacts(best_model, label_encoder, best_name, results):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model
    model_path = os.path.join(MODEL_DIR, 'combined_model.keras')
    best_model.save(model_path)
    print(f"  ✓ Model: {model_path}")

    # Labels
    lbl_path = os.path.join(MODEL_DIR, 'combined_label_encoder.npy')
    np.save(lbl_path, label_encoder.classes_)
    print(f"  ✓ Labels: {lbl_path}")

    # Meta JSON
    meta = {
        'model_type':       best_name,
        'labels':           label_encoder.classes_.tolist(),
        'sequence_length':  SEQUENCE_LENGTH,
        'n_hand_features':  N_HAND_FEATURES,
        'n_face_features':  N_FACE_FEATURES,
        'n_total_features': N_TOTAL_FEATURES,
        'test_accuracy':    float(results[best_name]['accuracy']),
    }
    meta_path = os.path.join(MODEL_DIR, 'combined_model_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Meta: {meta_path}")


# ══════════════════════════════════════════════════════ Main

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  VSL COMBINED TRAINER  (Biểu cảm mặt + Hành động tay)")
    print("="*65)

    # ── 1. Load
    print("\n[1/4] Đang tải dữ liệu...")
    X, y = load_combined_data(DATA_DIR)
    if len(X) == 0:
        print("\n✗ Không có data! Chạy combined_data_collector.py trước.")
        return

    print(f"\n  Tổng: {len(X)} mẫu, shape mỗi sample: {X.shape[1:]}")

    # ── 2. Encode + Split
    print("\n[2/4] Mã hóa nhãn và chia tập...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    n_classes = len(classes)
    print(f"  Classes ({n_classes}): {classes}")

    # Kiểm tra đủ điều kiện train
    label_counts  = {c: int(np.sum(y_enc == i)) for i, c in enumerate(classes)}
    min_samples   = min(label_counts.values())
    sparse_labels = [c for c, cnt in label_counts.items() if cnt < 5]

    if n_classes < 2:
        print("\n✗ Cần ít nhất 2 ký hiệu khác nhau để train!")
        print("  → Thu thập thêm ký hiệu khác bằng combined_data_collector.py")
        return

    if sparse_labels:
        print(f"\n⚠ Ký hiệu có ít hơn 5 mẫu: {sparse_labels}")
        print("  Khuyến nghị thu thập thêm để kết quả tốt hơn.")
        ans = input("  Tiếp tục train với dữ liệu hiện có? (y/n): ").strip().lower()
        if ans != "y":
            return

    if min_samples < 3:
        print(f"\n✗ Ký hiệu ít mẫu nhất chỉ có {min_samples} mẫu.")
        print("  Cần ít nhất 3 mẫu/ký hiệu để chia train/val/test.")
        print("  → Thu thập thêm bằng combined_data_collector.py")
        return

    # Tắt stratify nếu quá ít mẫu để tránh lỗi sklearn
    use_stratify = min_samples >= 5
    if not use_stratify:
        print("  ⚠ Ít mẫu, tắt stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if use_stratify else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
        stratify=y_train if use_stratify else None
    )
    print(f"  Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")

    # ── 3. Train all models
    print("\n[3/4] Huấn luyện các kiến trúc...")
    results    = {}
    histories  = []
    train_times = []
    trained_models = {}

    for name, builder in MODEL_BUILDERS.items():
        print(f"\n  ▶ {name}")
        model = builder(SEQUENCE_LENGTH, N_TOTAL_FEATURES, len(classes))
        model.summary(line_length=70)

        hist, t = train_model(model, X_train, y_train, X_val, y_val)
        ev = evaluate_model(model, X_test, y_test)

        results[name] = ev
        histories.append(hist)
        train_times.append(t)
        trained_models[name] = model

        print(f"  ✓ {name}: Acc={ev['accuracy']*100:.2f}%  "
              f"Loss={ev['loss']:.4f}  Time={t:.1f}s")

    # ── 4. Compare & Save
    print("\n[4/4] So sánh và lưu kết quả...")

    best_name  = max(results, key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_name]

    print(f"\n  🏆 Best: {best_name}  "
          f"Acc={results[best_name]['accuracy']*100:.2f}%")

    # Plots
    plot_comparison(results, os.path.join(RESULTS_DIR, 'model_comparison.png'))
    plot_training_curves(histories, list(MODEL_BUILDERS.keys()),
                         os.path.join(RESULTS_DIR, 'training_curves.png'))
    plot_confusion(y_test, results[best_name]['y_pred'], classes,
                   f'Confusion Matrix – {best_name}',
                   os.path.join(RESULTS_DIR, 'confusion_matrix.png'))

    # Classification report
    report = classification_report(y_test, results[best_name]['y_pred'],
                                   target_names=classes, digits=4)
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Best model: {best_name}\n\n{report}")
    print(f"  ✓ Report: {report_path}")
    print(report)

    # Summary
    summary_path = os.path.join(RESULTS_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nCOMBINED MODEL TRAINING SUMMARY\n" + "="*60 + "\n\n")
        f.write(f"{'Model':<25}{'Acc (%)':>12}{'Loss':>12}{'Time (s)':>12}\n")
        f.write("-"*60 + "\n")
        for (nm, ev), t in zip(results.items(), train_times):
            f.write(f"{nm:<25}{ev['accuracy']*100:>11.2f}%{ev['loss']:>12.4f}{t:>12.1f}\n")
        f.write(f"\nBest: {best_name}  Acc={results[best_name]['accuracy']*100:.2f}%\n")
    print(f"  ✓ Summary: {summary_path}")

    # Save all model weights
    for nm, mdl in trained_models.items():
        mdl.save(os.path.join(MODEL_DIR, f'combined_{nm}.keras'))

    # Save best model + meta
    save_artifacts(best_model, le, best_name, results)

    print("\n" + "="*65)
    print("  ✅ TRAINING HOÀN TẤT!")
    print(f"  Models: {MODEL_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()