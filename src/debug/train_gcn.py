"""
VSL Graph Convolutional Network (GCN) Trainer
Kiến trúc ST-GCN (Spatio-Temporal Graph Convolutional Networks)

FIXES:
- [BUG FIX] load_data_gcn: BỎ normalize vì data từ collect ĐÃ normalized rồi
            Normalize 2 lần → train/test distribution lệch hoàn toàn
- [BUG FIX] STGCN_Block: layers.Add() khởi tạo trong __init__, không phải call()
            Tạo layer mới mỗi forward pass → weights không được track đúng
- [BUG FIX] GraphConv: identity init + softmax normalize thay vì uniform
            Uniform random → gradient bất ổn lúc đầu train
- [IMPROVE] L2 regularization trên Dense cuối để tránh overfit với ít data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import glob
import matplotlib.pyplot as plt


# ==========================================
# DATA LOADER
# ==========================================
def load_data_gcn(dataset_dir):
    """
    Load data cho GCN.
    Pipeline: file .npy (30, 1659) → trích 75 điểm → (30, 75, 3)

    KHÔNG normalize ở đây — auto_collect_data.py đã normalize khi lưu.
    Normalize 2 lần làm lệch hoàn toàn với test (chỉ normalize 1 lần).

    75 điểm = Pose(33) + Left Hand(21) + Right Hand(21)
    Index trong vector 1659:
      - Pose:       0   → 98   (33*3)
      - Face:       99  → 1532 (478*3 — bỏ qua)
      - Left Hand:  1533→ 1595 (21*3)
      - Right Hand: 1596→ 1658 (21*3)
    """
    X, y = [], []
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    print(f"🔍 Tìm thấy {len(folders)} class: {sorted(folders)}")
    print("📂 Đang tải dữ liệu...")

    for sign_name in sorted(folders):
        sign_path = os.path.join(dataset_dir, sign_name)
        files     = glob.glob(os.path.join(sign_path, '*.npy'))
        count     = 0

        for f in files:
            try:
                seq = np.load(f)  # (30, 1659) — đã normalized từ collect

                if seq.shape != (30, 1659):
                    continue

                # Trích 75 điểm: Pose + 2 tay
                pose  = seq[:, 0:99]       # (30, 99)
                hands = seq[:, 1533:1659]  # (30, 126)

                skeleton          = np.concatenate([pose, hands], axis=1)  # (30, 225)
                skeleton_reshaped = skeleton.reshape(30, 75, 3)            # (30, 75, 3)

                X.append(skeleton_reshaped)
                y.append(sign_name)
                count += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {os.path.basename(f)}: {e}")

        print(f"  ✓ {sign_name}: {count} samples")

    return np.array(X), np.array(y)


# ==========================================
# MÔ HÌNH GCN
# ==========================================
class GraphConv(layers.Layer):
    """
    GCN với Adaptive Adjacency Matrix.

    Fix identity init:
    - Bắt đầu từ self-connection (mỗi node chỉ kết nối chính nó)
    - Dần học thêm kết nối với node lân cận
    - Uniform ngẫu nhiên 0-1 → gradient bất ổn ngay từ đầu

    Fix softmax normalize:
    - Đảm bảo tổng trọng số kết nối = 1 cho mỗi node
    - Ổn định gradient trong suốt quá trình train
    """
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels})
        return config

    def build(self, input_shape):
        self.nodes       = input_shape[2]
        self.in_channels = input_shape[3]

        self.A = self.add_weight(
            name="adjacency_matrix",
            shape=(self.nodes, self.nodes),
            initializer="identity",                    # ← Fix: identity thay vì uniform
            regularizer=keras.regularizers.l2(0.001),
            trainable=True
        )
        self.W = self.add_weight(
            name="weight_matrix",
            shape=(self.in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        # Normalize A bằng softmax để tổng trọng số mỗi node = 1
        A_norm = tf.nn.softmax(self.A, axis=-1)        # ← Fix: normalize A

        # Graph conv: (V,V) x (B,T,V,C) -> (B,T,V,C)
        x = tf.einsum('vw,btwc->btvc', A_norm, inputs)

        # Linear transform
        x = tf.matmul(x, self.W)
        return tf.nn.relu(x)


class STGCN_Block(layers.Layer):
    """
    Khối ST-GCN: GCN (spatial) + TCN (temporal) + Residual.

    Fix layers.Add():
    - Phải khởi tạo trong __init__, không phải call()
    - Nếu init trong call() → mỗi forward pass tạo Add layer mới
      → weights không được track → residual connection không học được
    """
    def __init__(self, out_channels, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.dropout_rate = dropout

        self.gcn        = GraphConv(out_channels)
        self.tcn        = layers.Conv2D(out_channels, kernel_size=(9, 1), padding='same', activation='relu')
        self.dropout    = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
        self.residual   = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')
        self.add        = layers.Add()  # ← Fix: khởi tạo 1 lần ở đây

    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels, 'dropout': self.dropout_rate})
        return config

    def call(self, inputs, training=None):
        x   = self.gcn(inputs)
        x   = self.tcn(x)
        x   = self.batch_norm(x, training=training)
        x   = self.dropout(x, training=training)
        res = self.residual(inputs)
        return self.add([x, res])  # ← Dùng self.add đã khởi tạo sẵn


def build_st_gcn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)  # (30, 75, 3)

    x = layers.BatchNormalization()(inputs)

    x = STGCN_Block(64,  name="stgcn_1")(x)
    x = STGCN_Block(64,  name="stgcn_2")(x)
    x = STGCN_Block(128, name="stgcn_3")(x)
    x = STGCN_Block(128, name="stgcn_4")(x)
    x = STGCN_Block(256, name="stgcn_5")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.01)  # Tránh overfit với ít data
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VSL_ST_GCN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==========================================
# HUẤN LUYỆN
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../data/raw')
    models_dir  = os.path.join(current_dir, '../models')
    results_dir = os.path.join(current_dir, '../results')
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    X, y = load_data_gcn(dataset_dir)

    if len(X) == 0:
        print("❌ Không tìm thấy dữ liệu hợp lệ!")
        return

    print(f"\n✅ Data shape: {X.shape}")  # (N, 30, 75, 3)

    # 2. Encode Labels
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_
    print(f"🏷️  Classes ({len(classes)}): {classes}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Model
    model = build_st_gcn_model(input_shape=(30, 75, 3), num_classes=len(classes))
    model.summary()

    # 5. Train
    print("\n🚀 Bắt đầu huấn luyện...")
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(models_dir, 'best_gcn_model.h5'),
        save_best_only=True, monitor='val_accuracy', verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=16,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 6. Evaluate
    print("\n📊 Đánh giá mô hình...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))

    # 7. Save
    np.save(os.path.join(models_dir, 'label_encoder_gcn.npy'), classes)
    print("✅ Đã lưu label encoder.")

    # 8. Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gcn_training_history.png'))
    print("✅ Đã lưu biểu đồ.")


if __name__ == '__main__':
    main()
