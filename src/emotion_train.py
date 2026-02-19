"""
VSL Emotion Model Trainer
Train ML classifier để nhận diện cảm xúc từ facial landmarks
"""

import numpy as np
import os
import glob
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, '..', 'data', 'emotion_raw')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
RESULT_DIR = os.path.join(BASE_DIR, '..', 'results', 'emotion')

EMOTIONS  = ['happy', 'sad', 'angry', 'surprise', 'worried', 'disgust', 'neutral']

FEATURE_NAMES = [
    'left_ear', 'right_ear', 'avg_ear',
    'mar',
    'left_brow', 'right_brow', 'avg_brow',
    'nose_chin_dist',
    'eye_dist'
]


# ================================================================= Load data

def load_emotion_data(data_dir: str):
    X, y = [], []

    print(f"\n📂 Loading emotion data from: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục: {data_dir}")
        print("   Hãy chạy emotion_data_collector.py trước!")
        return np.array([]), np.array([])

    label_counts = {}

    for emotion in EMOTIONS:
        emo_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emo_dir):
            continue

        files = glob.glob(os.path.join(emo_dir, '*.npy'))
        if not files:
            print(f"  ⚠️  '{emotion}': 0 mẫu (bỏ qua)")
            continue

        for f in files:
            try:
                feat = np.load(f)
                if feat.shape == (9,):
                    X.append(feat)
                    y.append(emotion)
                else:
                    print(f"  ⚠️  Bỏ qua shape lạ {feat.shape}: {f}")
            except Exception as e:
                print(f"  ❌ Lỗi đọc {f}: {e}")

        label_counts[emotion] = len(files)

    print("\n📊 Thống kê:")
    for emo, cnt in label_counts.items():
        bar = "█" * (cnt // 5)
        print(f"  {emo:<10} {cnt:>4} mẫu  {bar}")

    total = len(X)
    print(f"\n  Tổng: {total} mẫu, {len(label_counts)} cảm xúc")

    if total < 50:
        print("\n⚠️  Cảnh báo: Dữ liệu quá ít (<50 mẫu). Kết quả có thể không tốt.")
        print("   Khuyến nghị: >=100 mẫu mỗi cảm xúc.")

    return np.array(X), np.array(y)


# ================================================================= Train

def train_and_compare(X, y):
    """Train nhiều model và so sánh"""

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ---- Define models ----
    models = {
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                learning_rate_init=0.001,
            ))
        ]),
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42,
            ))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            ))
        ]),
    }

    results = {}

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    for name, pipeline in models.items():
        print(f"\n🚀 Training {name}...")

        # Cross-val trên train set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train đầy đủ
        pipeline.fit(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        y_pred   = pipeline.predict(X_test)

        print(f"   Test Accuracy: {test_acc:.4f}")

        results[name] = {
            'pipeline': pipeline,
            'cv_mean': cv_scores.mean(),
            'cv_std':  cv_scores.std(),
            'test_acc': test_acc,
            'y_pred':   y_pred,
        }

    # ---- Find best ----
    best_name = max(results, key=lambda k: results[k]['test_acc'])
    best = results[best_name]

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<20} {'CV Acc':<15} {'Test Acc'}")
    print(f"  {'-'*50}")
    for name, r in results.items():
        mark = "🏆" if name == best_name else "  "
        print(f"  {mark} {name:<18} {r['cv_mean']:.4f}±{r['cv_std']:.4f}   {r['test_acc']:.4f}")

    print(f"\n🏆 Best Model: {best_name}  (Test Acc = {best['test_acc']:.4f})")

    return results, best_name, label_encoder, X_test, y_test


# ================================================================= Visualize

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=classes, yticklabels=classes,
        cmap='Blues', linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_model_comparison(results, save_path):
    names = list(results.keys())
    test_accs = [results[n]['test_acc'] * 100 for n in names]
    cv_means  = [results[n]['cv_mean']  * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cv_means,  width, label='CV Accuracy',   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',  color='coral',     alpha=0.8)

    # Highlight best
    best_idx = np.argmax(test_accs)
    bars2[best_idx].set_color('gold')
    bars2[best_idx].set_edgecolor('darkgoldenrod')
    bars2[best_idx].set_linewidth(2)

    ax.set_title('Emotion Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_feature_importance(model_pipeline, save_path):
    """Vẽ feature importance nếu model hỗ trợ"""
    clf = model_pipeline.named_steps.get('clf')
    if not hasattr(clf, 'feature_importances_'):
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)),
            importances[indices],
            color='steelblue', edgecolor='navy')
    plt.xticks(range(len(importances)),
               [FEATURE_NAMES[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Feature Importance (Best Model)', fontsize=13, fontweight='bold')
    plt.ylabel('Importance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ================================================================= Save

def save_model(pipeline, label_encoder, model_dir, best_name):
    os.makedirs(model_dir, exist_ok=True)

    # Save pipeline (scaler + classifier)
    model_path   = os.path.join(model_dir, 'emotion_classifier.pkl')
    encoder_path = os.path.join(model_dir, 'emotion_label_encoder.pkl')
    meta_path    = os.path.join(model_dir, 'emotion_model_meta.json')

    joblib.dump(pipeline,       model_path)
    joblib.dump(label_encoder,  encoder_path)

    meta = {
        'model_type':   best_name,
        'emotions':     label_encoder.classes_.tolist(),
        'features':     FEATURE_NAMES,
        'n_features':   len(FEATURE_NAMES),
        'model_file':   'emotion_classifier.pkl',
        'encoder_file': 'emotion_label_encoder.pkl',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Model saved:")
    print(f"   {model_path}")
    print(f"   {encoder_path}")
    print(f"   {meta_path}")


# ================================================================= Main

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    print("\n" + "="*60)
    print("VSL EMOTION MODEL TRAINER")
    print("="*60)

    # Load data
    X, y = load_emotion_data(DATA_DIR)

    if len(X) == 0:
        print("\n❌ Không có data! Chạy emotion_data_collector.py trước.")
        return

    # Train & compare
    results, best_name, label_encoder, X_test, y_test = train_and_compare(X, y)

    # Classification report
    best_pipeline = results[best_name]['pipeline']
    y_pred        = results[best_name]['y_pred']
    classes       = label_encoder.classes_

    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print("\n📄 Classification Report (Best Model):")
    print(report)

    report_path = os.path.join(RESULT_DIR, 'emotion_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Best Model: {best_name}\n\n")
        f.write(report)
    print(f"✓ Saved: {report_path}")

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, classes,
        f'Confusion Matrix - {best_name}',
        os.path.join(RESULT_DIR, 'emotion_confusion_matrix.png')
    )

    # Model comparison
    plot_model_comparison(
        results,
        os.path.join(RESULT_DIR, 'emotion_model_comparison.png')
    )

    # Feature importance
    plot_feature_importance(
        best_pipeline,
        os.path.join(RESULT_DIR, 'emotion_feature_importance.png')
    )

    # Save best model
    save_model(best_pipeline, label_encoder, MODEL_DIR, best_name)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Model: {best_name}")
    print(f"   Test Accuracy: {results[best_name]['test_acc']*100:.2f}%")
    print(f"   Results: {RESULT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()