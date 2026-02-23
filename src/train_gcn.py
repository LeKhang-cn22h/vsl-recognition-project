"""
VSL ST-GCN + GRU Trainer

FIX:
- [FIX] Thay Lambda layer bằng NodePool custom layer
        Lambda bị Keras chặn khi load vì lý do bảo mật (arbitrary code execution)
        Custom layer với get_config() → load/save an toàn, không cần safe_mode=False
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
    Load: file .npy (30, 1659) → trích 75 điểm → (30, 75, 3)
    KHÔNG normalize — data đã normalize hierarchical từ collect.
    """
    X, y    = [], []
    folders = [f for f in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, f))]

    print(f"🔍 Tìm thấy {len(folders)} class: {sorted(folders)}")

    for sign_name in sorted(folders):
        sign_path = os.path.join(dataset_dir, sign_name)
        files     = glob.glob(os.path.join(sign_path, '*.npy'))
        count     = 0

        for f in files:
            try:
                seq = np.load(f)
                if seq.shape != (30, 1659):
                    continue

                pose     = seq[:, 0:99]
                hands    = seq[:, 1533:1659]
                skeleton = np.concatenate([pose, hands], axis=1)

                X.append(skeleton.reshape(30, 75, 3))
                y.append(sign_name)
                count += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi {os.path.basename(f)}: {e}")

        print(f"  ✓ {sign_name}: {count} samples")

    return np.array(X), np.array(y)


# ==========================================
# CUSTOM LAYERS
# ==========================================
class GraphConv(layers.Layer):
    """GCN với Adaptive Adjacency Matrix (identity init + softmax normalize)."""
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
            initializer="identity",
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
        A_norm = tf.nn.softmax(self.A, axis=-1)
        x = tf.einsum('vw,btwc->btvc', A_norm, inputs)
        x = tf.matmul(x, self.W)
        return tf.nn.relu(x)


class STGCN_Block(layers.Layer):
    """ST-GCN Block: GCN (spatial) + Conv2D (temporal) + Residual."""
    def __init__(self, out_channels, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.dropout_rate = dropout
        self.gcn        = GraphConv(out_channels)
        self.tcn        = layers.Conv2D(out_channels, kernel_size=(9, 1),
                                        padding='same', activation='relu')
        self.dropout    = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
        self.residual   = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')
        self.add        = layers.Add()

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
        return self.add([x, res])


class NodePool(layers.Layer):
    """
    Pool theo chiều Node (axis=2), GIỮ chiều Time.
    (B, T, V, C) → (B, T, C) bằng mean theo V.

    Thay thế Lambda layer để tránh lỗi Keras security:
    'Requested the deserialization of a Lambda layer... disallowed by default'
    Custom layer có get_config() → serialize/deserialize an toàn.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        # inputs: (B, T, V, C) → output: (B, T, C)
        return tf.reduce_mean(inputs, axis=2)


# ==========================================
# MODEL
# ==========================================
def build_st_gcn_model(input_shape, num_classes):
    """
    Pipeline:
      (B, 30, 75, 3)
        → BatchNorm
        → STGCN x5        → (B, 30, 75, 256)
        → NodePool         → (B, 30, 256)      [mean theo node, GIỮ time]
        → GRU(128)         → (B, 128)           [học temporal order]
        → Dropout → Dense  → (B, num_classes)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.BatchNormalization()(inputs)

    x = STGCN_Block(64,  name="stgcn_1")(x)
    x = STGCN_Block(64,  name="stgcn_2")(x)
    x = STGCN_Block(128, name="stgcn_3")(x)
    x = STGCN_Block(128, name="stgcn_4")(x)
    x = STGCN_Block(256, name="stgcn_5")(x)
    # (B, 30, 75, 256)

    x = NodePool(name="node_pool")(x)
    # (B, 30, 256)

    x = layers.GRU(128, return_sequences=False, name="temporal_gru")(x)
    # (B, 128)

    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="classifier"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VSL_STGCN_GRU")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==========================================
# TRAINING
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../data/raw')
    models_dir  = os.path.join(current_dir, '../models')
    results_dir = os.path.join(current_dir, '../results')
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    X, y = load_data_gcn(dataset_dir)
    if len(X) == 0:
        print("❌ Không tìm thấy dữ liệu!")
        return
    print(f"\n✅ Data shape: {X.shape}")

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_
    print(f"🏷️  Classes ({len(classes)}): {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_st_gcn_model(input_shape=(30, 75, 3), num_classes=len(classes))
    model.summary()

    print("\n🚀 Bắt đầu huấn luyện ST-GCN + GRU...")
    model_save_path = os.path.join(models_dir, 'best_gcn_model.h5')

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path, save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=16,
        callbacks=callbacks
    )

    print("\n📊 Đánh giá...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))

    np.save(os.path.join(models_dir, 'label_encoder_gcn.npy'), classes)
    print("✅ Đã lưu label encoder.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gcn_training_history.png'))
    print("✅ Đã lưu biểu đồ.")


if __name__ == '__main__':
    main()
