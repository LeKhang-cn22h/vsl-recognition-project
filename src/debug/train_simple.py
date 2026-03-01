"""
VSL Multi-Model Trainer & Comparator
Huấn luyện và so sánh nhiều kiến trúc mô hình khác nhau
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import os
import glob
from keras.callbacks import EarlyStopping

import json

def load_data(dataset_dir):
    """Load data và tự động lọc các file sai kích thước"""
    X_temp, y_temp = [], []
    
    print(f"📂 Đang quét data tại: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục {dataset_dir}")
        return np.array([]), np.array([])

    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    
    if not folders:
        print("❌ Không tìm thấy folder nào trong data/raw!")
        return np.array([]), np.array([])

    # Bước 1: Quét toàn bộ để thống kê shape phổ biến nhất
    shape_counter = {}
    valid_files = []

    print("🔍 Đang phân tích cấu trúc dữ liệu...")
    for sign_name in folders:
        sign_path = os.path.join(dataset_dir, sign_name)
        sample_files = glob.glob(os.path.join(sign_path, '*.npy'))
        
        for f in sample_files:
            try:
                seq = np.load(f)
                shape = seq.shape
                # Chỉ quan tâm sequence length = 30
                if shape[0] == 30:
                    if shape not in shape_counter:
                        shape_counter[shape] = 0
                    shape_counter[shape] += 1
                    valid_files.append((f, sign_name, seq))
            except:
                pass

    if not shape_counter:
        print("❌ Không tìm thấy file data hợp lệ (len=30)!")
        return np.array([]), np.array([])

    # Tìm shape phổ biến nhất (ví dụ: (30, 1659) cho Holistic hoặc (30, 126) cho Hand)
    target_shape = max(shape_counter, key=shape_counter.get)
    print(f"✅ Shape chuẩn được chọn: {target_shape} (chiếm {shape_counter[target_shape]} mẫu)")
    
    if len(shape_counter) > 1:
        print(f"⚠️ Cảnh báo: Phát hiện dữ liệu lẫn lộn {shape_counter}. Đang lọc bỏ dữ liệu rác...")

    # Bước 2: Chỉ lấy data đúng target_shape
    for f_path, label, seq in valid_files:
        if seq.shape == target_shape:
            X_temp.append(seq)
            y_temp.append(label)
    
    return np.array(X_temp), np.array(y_temp)


# ============ ĐỊNH NGHĨA CÁC MÔ HÌNH ============

def build_simple_lstm(sequence_length, n_features, n_classes):
    """Model 1: Simple LSTM - Mô hình gốc đơn giản"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name='Simple_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_bidirectional_lstm(sequence_length, n_features, n_classes):
    """Model 2: Bidirectional LSTM - Học từ cả 2 chiều"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name='Bidirectional_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_gru_model(sequence_length, n_features, n_classes):
    """Model 3: GRU - Nhanh hơn LSTM, ít tham số hơn"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name='GRU_Model')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_lstm(sequence_length, n_features, n_classes):
    """Model 4: CNN-LSTM Hybrid - CNN trích xuất đặc trưng, LSTM học chuỗi"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        
        # CNN layers để trích xuất đặc trưng
        layers.Reshape((sequence_length, n_features, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        
        # Reshape lại cho LSTM
        layers.Reshape((sequence_length // 4, -1)),
        
        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name='CNN_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_attention_lstm(sequence_length, n_features, n_classes):
    """Model 5: LSTM with Attention - Tập trung vào phần quan trọng"""
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # LSTM layer
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=4, 
        key_dim=32
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Another LSTM
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Attention_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_deep_lstm(sequence_length, n_features, n_classes):
    """Model 6: Deep Stacked LSTM - LSTM nhiều lớp hơn"""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name='Deep_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Dictionary chứa tất cả các model builders
MODEL_BUILDERS = {
    'simple_lstm': build_simple_lstm,
    'bidirectional_lstm': build_bidirectional_lstm,
    'gru': build_gru_model,
    'cnn_lstm': build_cnn_lstm,
    'attention_lstm': build_attention_lstm,
    'deep_lstm': build_deep_lstm,
}


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """Huấn luyện một mô hình"""
    print(f"\n{'='*60}")
    print(f"🚀 Đang huấn luyện: {model_name}")
    print(f"{'='*60}")
    
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    return history, training_time


def evaluate_model(model, X_test, y_test, model_name):
    """Đánh giá mô hình"""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    return {
        'name': model_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'y_pred': y_pred
    }


def plot_model_comparison(results, save_path):
    """So sánh accuracy của các mô hình"""
    model_names = [r['name'] for r in results]
    accuracies = [r['test_accuracy'] * 100 for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='steelblue', edgecolor='navy', linewidth=2)
    
    # Tô màu cho model tốt nhất
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Average: {np.mean(accuracies):.2f}%')
    
    plt.title('Model Comparison - Test Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên mỗi cột
    for i, (name, acc) in enumerate(zip(model_names, accuracies)):
        color = 'darkgoldenrod' if i == best_idx else 'black'
        weight = 'bold' if i == best_idx else 'normal'
        plt.text(i, acc + 1, f'{acc:.2f}%', 
                ha='center', va='bottom', fontweight=weight, 
                fontsize=11, color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ so sánh: {save_path}")
    plt.close()


def plot_training_comparison(histories, model_names, save_path):
    """So sánh quá trình training của các mô hình"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Accuracy plot
    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = colors[i % len(colors)]
        axes[0].plot(history.history['val_accuracy'], 
                    label=name, linewidth=2, color=color, alpha=0.8)
    
    axes[0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = colors[i % len(colors)]
        axes[1].plot(history.history['val_loss'], 
                    label=name, linewidth=2, color=color, alpha=0.8)
    
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ training comparison: {save_path}")
    plt.close()


def save_results_summary(results, training_times, save_path):
    """Lưu bảng tổng kết kết quả"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Model Name':<25} {'Test Acc (%)':<15} {'Test Loss':<15} {'Train Time (s)':<15}\n")
        f.write("-"*80 + "\n")
        
        for result, train_time in zip(results, training_times):
            f.write(f"{result['name']:<25} "
                   f"{result['test_accuracy']*100:>13.2f}% "
                   f"{result['test_loss']:>14.4f} "
                   f"{train_time:>14.1f}\n")
        
        f.write("\n" + "="*80 + "\n")
        
        # Best model
        best_idx = np.argmax([r['test_accuracy'] for r in results])
        best_model = results[best_idx]
        
        f.write(f"\n🏆 BEST MODEL: {best_model['name']}\n")
        f.write(f"   - Accuracy: {best_model['test_accuracy']*100:.2f}%\n")
        f.write(f"   - Loss: {best_model['test_loss']:.4f}\n")
        f.write(f"   - Training Time: {training_times[best_idx]:.1f}s\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Đã lưu bảng tổng kết: {save_path}")
    
    # In ra console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model Name':<25} {'Test Acc (%)':<15} {'Test Loss':<15} {'Train Time (s)':<15}")
    print("-"*80)
    for result, train_time in zip(results, training_times):
        print(f"{result['name']:<25} "
              f"{result['test_accuracy']*100:>13.2f}% "
              f"{result['test_loss']:>14.4f} "
              f"{train_time:>14.1f}")
    print("\n🏆 BEST MODEL: " + best_model['name'])
    print(f"   Accuracy: {best_model['test_accuracy']*100:.2f}%")

def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4
    )
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n📄 Classification Report:")
    print(report)

def save_top_confusions(y_true, y_pred, classes, save_path, top_k=5):
    cm = confusion_matrix(y_true, y_pred)
    confusions = []

    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append((classes[i], classes[j], cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Top Confused Gesture Pairs\n")
        f.write("="*40 + "\n")
        for a, b, c in confusions[:top_k]:
            f.write(f"{a} → {b}: {c} samples\n")

def main():
    print("\n" + "="*60)
    print("VSL MULTI-MODEL TRAINER & COMPARATOR")
    print("="*60)
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../data/raw')
    models_dir = os.path.join(current_dir, '../models')
    results_dir = os.path.join(current_dir, '../results/comparison')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    X, y = load_data(dataset_dir)
    
    if len(X) == 0:
        print("\n❌ KHÔNG CÓ DATA! Chạy auto_collect_data.py trước.")
        return
    
    print(f"✅ Tổng cộng: {len(X)} mẫu")
    
    # Encode labels
    print("\n[2/4] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    print(f"✅ Đã mã hóa {len(classes)} nhãn: {classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Val set:   {len(X_val)} samples")
    print(f"✓ Test set:  {len(X_test)} samples")
    
    # Train all models
    print("\n[3/4] Training all models...")
    print("="*60)
    
    results = []
    histories = []
    training_times = []
    models = []
    
    for model_key, model_builder in MODEL_BUILDERS.items():
        print(f"\n>>> Training {model_key}...")
        
        # Build model
        model = model_builder(
            sequence_length=X.shape[1],
            n_features=X.shape[2],
            n_classes=len(classes)
        )
        
        # Train
        history, train_time = train_model(
            model, X_train, y_train, X_val, y_val, model_key
        )
        
        # Evaluate
        result = evaluate_model(model, X_test, y_test, model_key)
        
        # Save
        results.append(result)
        histories.append(history)
        training_times.append(train_time)
        models.append(model)
        
        print(f"✓ {model_key}: Acc={result['test_accuracy']*100:.2f}%, "
              f"Loss={result['test_loss']:.4f}, Time={train_time:.1f}s")
    
    # Compare and visualize
    print("\n[4/4] Comparing and visualizing results...")
    print("="*60)
    
    model_names = [r['name'] for r in results]
    
    # Plot comparison
    plot_model_comparison(
        results,
        os.path.join(results_dir, 'model_accuracy_comparison.png')
    )
    
    plot_training_comparison(
        histories, model_names,
        os.path.join(results_dir, 'training_progress_comparison.png')
    )
    
    save_results_summary(
        results, training_times,
        os.path.join(results_dir, 'comparison_summary.txt')
    )
    
    # Save best model
    best_idx = np.argmax([r['test_accuracy'] for r in results])
    best_model = models[best_idx]
    best_model_name = results[best_idx]['name']

    best_result = results[best_idx]

    save_classification_report(
        y_test,
        best_result['y_pred'],
        classes,
        os.path.join(results_dir, 'classification_report_best_model.txt')
    )

    save_top_confusions(
        y_test,
        best_result['y_pred'],
        classes,
        os.path.join(results_dir, 'top_confused_pairs.txt')
    )
    
    # 7. Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    #  Độ chính xác trên dữ liệu chưa từng thấy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save labels
    np.save(os.path.join(models_dir, 'label_encoder.npy'), classes)
    
    print("\n" + "="*60)
    print("✅ TRAINING & COMPARISON COMPLETE!")
    print("="*60)
    print(f"\n📁 Kết quả được lưu tại: {results_dir}")
    print(f"📁 Models được lưu tại: {models_dir}")
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results[best_idx]['test_accuracy']*100:.2f}%")
    print(f"   Path: {best_model_path}")

if __name__ == '__main__':
    main()