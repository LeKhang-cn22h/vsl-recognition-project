"""
auto_label_emotion.py
=====================================================================
Tự động gán emotion cho tất cả video trong dataset bằng EfficientNet-B2.

Flow:
  1. Với mỗi folder label (vd: data/videos/train/ai/)
  2. Đọc tất cả video trong folder → sample frames
  3. Predict emotion từng frame bằng EfficientNet-B2
  4. Tính trung bình probability toàn folder → lấy emotion cao nhất
  5. Gán emotion đó cho TẤT CẢ video trong folder (lưu vào .json)

Mapping CNN → video_to_npy:
  angry    → angry
  cry      → sad
  disgust  → disgust
  neutral  → neutral
  scare    → fear
  smile    → happy
  surprise → surprise

Chạy:
  python auto_label_emotion.py                     # tất cả splits (tự skip đã gán)
  python auto_label_emotion.py --split train       # chỉ train
  python auto_label_emotion.py --split train val   # train + val
  python auto_label_emotion.py --label ai lo_so    # chỉ label cụ thể
  python auto_label_emotion.py --dry-run           # preview không gán
  python auto_label_emotion.py --frames 10         # số frame/video (default 8)
  python auto_label_emotion.py --no-skip           # ghi đè kể cả đã có emotion
  python auto_label_emotion.py --conf-thresh 0.8   # ngưỡng confidence (default 0.7)
  python auto_label_emotion.py --no-interactive    # tự động bỏ qua conf thấp, không hỏi

  # Gỡ nhãn các file đã gán có confidence thấp:
  python auto_label_emotion.py --unlabel-low-conf              # gỡ conf < 0.7
  python auto_label_emotion.py --unlabel-low-conf --conf-thresh 0.8  # gỡ conf < 0.8
  python auto_label_emotion.py --unlabel-low-conf --dry-run    # preview trước khi gỡ
"""

import os
import cv2
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# ══════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════

MODEL_PATH      = r'checkpoints/emotion_cnn_best.pth'
VIDEO_DIR       = r'datamlp'
CONF_THRESH_DEF = 0.70          # ngưỡng confidence mặc định

EMOTION_MAP = {
    "angry":    "angry",
    "cry":      "sad",
    "disgust":  "disgust",
    "neutral":  "neutral",
    "scare":    "fear",
    "smile":    "happy",
    "surprise": "surprise",
}

VIDEO_TO_NPY_EMOTIONS = {
    "angry": 0, "disgust": 1, "fear": 2,
    "happy": 3, "sad": 4,    "surprise": 5, "neutral": 6,
}

ALL_MAPPED_EMOTIONS = sorted(VIDEO_TO_NPY_EMOTIONS.keys())

SPLITS     = ['train', 'val', 'test']
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = efficientnet_b2(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path: str):
    print(f"\n  Loading model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    ckpt     = torch.load(model_path, map_location=DEVICE)
    emotions = ckpt.get('emotions',
                        ['angry', 'cry', 'disgust', 'neutral', 'scare', 'smile', 'surprise'])
    model    = EmotionCNN(num_classes=len(emotions)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"  Device  : {DEVICE}")
    print(f"  Epoch   : {ckpt.get('epoch', '?')}")
    print(f"  Bal Acc : {ckpt.get('bal_acc', 0)*100:.1f}%")
    print(f"  Emotions: {emotions}")
    return model, emotions


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def extract_face(frame_bgr: np.ndarray) -> np.ndarray:
    """Tách mặt từ frame. Fallback → resize frame gốc nếu không thấy mặt."""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        pad = int(0.15 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame_bgr.shape[1], x + w + pad)
        y2  = min(frame_bgr.shape[0], y + h + pad)
        face = frame_bgr[y1:y2, x1:x2]
        if face.size > 0:
            return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(frame_bgr, (224, 224))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def sample_frames(video_path: str, n_frames: int = 8) -> list:
    """Sample đều n_frames từ video. Trả về list RGB numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(extract_face(frame))
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_frames(frames: list, model: EmotionCNN) -> np.ndarray:
    """Predict emotion cho list frames → average probability vector (n_emotions,)."""
    if not frames:
        return None
    batch  = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
    logits = model(batch)
    probs  = torch.softmax(logits, dim=1)
    return probs.cpu().numpy().mean(axis=0)


def predict_folder_videos(videos: list, model: EmotionCNN,
                           cnn_emotions: list, n_frames: int = 8) -> dict:
    """
    Predict emotion cho list video paths.
    Average probability toàn bộ → emotion cao nhất.
    """
    if not videos:
        return None

    all_probs    = []
    total_frames = 0

    for vpath in videos:
        frames = sample_frames(str(vpath), n_frames)
        if not frames:
            continue
        prob = predict_frames(frames, model)
        if prob is not None:
            all_probs.append(prob)
            total_frames += len(frames)

    if not all_probs:
        return None

    avg_probs      = np.mean(all_probs, axis=0)
    best_idx       = int(np.argmax(avg_probs))
    emotion_cnn    = cnn_emotions[best_idx]
    emotion_mapped = EMOTION_MAP.get(emotion_cnn, emotion_cnn)

    return {
        'emotion_cnn':    emotion_cnn,
        'emotion_mapped': emotion_mapped,
        'confidence':     float(avg_probs[best_idx]),
        'avg_probs':      {cnn_emotions[i]: float(avg_probs[i])
                           for i in range(len(cnn_emotions))},
        'n_videos':       len(videos),
        'n_frames_used':  total_frames,
    }


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE PROMPT — low confidence
# ══════════════════════════════════════════════════════════════════

def prompt_low_conf(split: str, label: str,
                    emotion_mapped: str, confidence: float,
                    top3_str: str, conf_thresh: float) -> str | None:
    """
    Hỏi user khi confidence < conf_thresh.

    Returns:
        str  → emotion đã chọn (có thể là gán tay)
        None → bỏ qua (không gán)
    """
    print(f"\n    ⚠️  Confidence thấp ({confidence*100:.1f}% < {conf_thresh*100:.0f}%)")
    print(f"    CNN gợi ý : '{emotion_mapped}'")
    print(f"    Top probs : {top3_str}")
    print(f"\n    Chọn hành động cho [{split}/{label}]:")
    print(f"      [1] Giữ '{emotion_mapped}' (chấp nhận kết quả CNN)")
    print(f"      [2] Bỏ qua (không gán emotion cho folder này)")
    print(f"      [3] Gán tay (tự chọn emotion)")

    while True:
        try:
            choice = input("    → Nhập lựa chọn (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n    ⚠️  Bỏ qua do interrupt.")
            return None

        if choice == '1':
            return emotion_mapped

        elif choice == '2':
            print(f"    ⏭️  Bỏ qua [{split}/{label}]")
            return None

        elif choice == '3':
            print(f"\n    Danh sách emotion hợp lệ:")
            for i, e in enumerate(ALL_MAPPED_EMOTIONS, 1):
                print(f"      [{i}] {e}")
            while True:
                try:
                    raw = input("    → Nhập tên hoặc số thứ tự emotion: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n    ⚠️  Bỏ qua do interrupt.")
                    return None

                # Nhập số thứ tự
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(ALL_MAPPED_EMOTIONS):
                        chosen = ALL_MAPPED_EMOTIONS[idx]
                        print(f"    ✏️  Đã chọn gán tay: '{chosen}'")
                        return chosen
                    else:
                        print(f"    ❌ Số không hợp lệ. Nhập 1–{len(ALL_MAPPED_EMOTIONS)}.")
                # Nhập tên
                elif raw in ALL_MAPPED_EMOTIONS:
                    print(f"    ✏️  Đã chọn gán tay: '{raw}'")
                    return raw
                else:
                    print(f"    ❌ '{raw}' không hợp lệ. Chọn: {ALL_MAPPED_EMOTIONS}")
        else:
            print("    ❌ Nhập 1, 2 hoặc 3.")


# ══════════════════════════════════════════════════════════════════
# JSON HELPERS
# ══════════════════════════════════════════════════════════════════

def get_existing_emotion(video_path: str):
    """Đọc emotion đã gán từ .json kế bên video. Trả về None nếu chưa có."""
    meta_path = Path(video_path).with_suffix('.json')
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return data.get('emotion', None)
    except Exception:
        return None


def get_existing_confidence(video_path: str) -> float | None:
    """Đọc emotion_confidence từ .json. Trả về None nếu chưa có hoặc lỗi."""
    meta_path = Path(video_path).with_suffix('.json')
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return data.get('emotion_confidence', None)
    except Exception:
        return None


def count_labeled(videos: list) -> tuple:
    """Đếm (đã gán, chưa gán). Returns (labeled, unlabeled)."""
    labeled   = sum(1 for v in videos if get_existing_emotion(str(v)) is not None)
    unlabeled = len(videos) - labeled
    return labeled, unlabeled


def save_emotion_json(video_path: str, emotion_mapped: str,
                      prediction_info: dict, manual: bool = False) -> str:
    """Lưu emotion vào .json kế bên video."""
    meta_path = str(Path(video_path).with_suffix('.json'))

    data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                data = json.load(f)
        except Exception:
            pass

    data['emotion']            = emotion_mapped
    data['emotion_id']         = VIDEO_TO_NPY_EMOTIONS.get(emotion_mapped, 0)
    data['emotion_cnn_raw']    = prediction_info.get('emotion_cnn', 'manual')
    data['emotion_confidence'] = round(prediction_info.get('confidence', 0.0), 4)
    data['emotion_source']     = 'manual' if manual else 'auto_label_emotion.py'
    data['emotion_updated']    = datetime.now().isoformat()

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return meta_path


def remove_emotion_json(video_path: str) -> bool:
    """
    Xóa các trường emotion khỏi .json kế bên video.
    Nếu .json rỗng sau khi xóa → xóa luôn file.
    Returns True nếu thành công.
    """
    meta_path = Path(video_path).with_suffix('.json')
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            data = json.load(f)
    except Exception:
        return False

    emotion_keys = [
        'emotion', 'emotion_id', 'emotion_cnn_raw',
        'emotion_confidence', 'emotion_source', 'emotion_updated',
    ]
    changed = any(k in data for k in emotion_keys)
    for k in emotion_keys:
        data.pop(k, None)

    if not data:
        meta_path.unlink()          # file rỗng → xóa hẳn
    else:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return changed


# ══════════════════════════════════════════════════════════════════
# UNLABEL LOW-CONFIDENCE
# ══════════════════════════════════════════════════════════════════

def unlabel_low_conf(video_dir: str, splits: list, label_filter: list,
                     conf_thresh: float, dry_run: bool):
    """
    Duyệt toàn bộ .json, tìm file có emotion_confidence < conf_thresh
    và gỡ nhãn (xóa các trường emotion).
    """
    video_dir = Path(video_dir)
    removed   = []
    skipped   = []

    print(f"\n{'='*62}")
    print(f"  UNLABEL LOW-CONFIDENCE  (ngưỡng: {conf_thresh*100:.0f}%)")
    print(f"{'='*62}")

    for split in splits:
        split_dir = video_dir / split
        if not split_dir.exists():
            continue

        label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if label_filter:
            label_dirs = [d for d in label_dirs if d.name in label_filter]

        for label_dir in label_dirs:
            label  = label_dir.name
            videos = sorted([p for p in label_dir.iterdir()
                              if p.suffix.lower() in VIDEO_EXTS])

            folder_removed = []
            folder_skipped = []

            for vpath in videos:
                conf = get_existing_confidence(str(vpath))
                if conf is None:
                    continue                         # chưa có nhãn → bỏ qua

                emotion = get_existing_emotion(str(vpath))

                if conf < conf_thresh:
                    folder_removed.append((vpath, emotion, conf))
                else:
                    folder_skipped.append((vpath, emotion, conf))

            if not folder_removed:
                continue

            print(f"\n  [{split}/{label}]  "
                  f"{len(folder_removed)} file cần gỡ  |  "
                  f"{len(folder_skipped)} file giữ lại")

            for vpath, emotion, conf in folder_removed:
                tag = "[DRY RUN] " if dry_run else ""
                print(f"    {tag}🗑️  {vpath.name}  "
                      f"→ '{emotion}'  conf={conf*100:.1f}%")
                if not dry_run:
                    remove_emotion_json(str(vpath))

            removed.extend(folder_removed)
            skipped.extend(folder_skipped)

    # Summary
    print(f"\n{'='*62}")
    print(f"  TỔNG KẾT UNLABEL")
    print(f"{'='*62}")
    print(f"  Đã gỡ nhãn : {len(removed)} file  (conf < {conf_thresh*100:.0f}%)")
    print(f"  Giữ lại    : {len(skipped)} file  (conf ≥ {conf_thresh*100:.0f}%)")

    if dry_run:
        print(f"\n  ⚠️  DRY RUN — chưa xóa gì cả!")
        print(f"  Chạy lại không có --dry-run để thực sự gỡ nhãn.")
    else:
        if removed:
            print(f"\n  ✅ Đã gỡ nhãn xong.")
            print(f"  Bạn có thể chạy lại auto_label_emotion.py để gán lại.")
        else:
            print(f"\n  ✅ Không có file nào cần gỡ.")
    print(f"{'='*62}\n")


# ══════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════

def process_all(video_dir: str, splits: list, label_filter: list,
                n_frames: int, dry_run: bool, skip_labeled: bool,
                conf_thresh: float, interactive: bool,
                model: EmotionCNN, cnn_emotions: list):

    video_dir = Path(video_dir)
    summary   = []

    for split in splits:
        split_dir = video_dir / split
        if not split_dir.exists():
            print(f"\n  ⚠️  Split không tồn tại: {split_dir} — bỏ qua")
            continue

        label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if label_filter:
            label_dirs = [d for d in label_dirs if d.name in label_filter]

        if not label_dirs:
            print(f"\n  ⚠️  [{split}] Không có label nào phù hợp")
            continue

        print(f"\n{'='*62}")
        print(f"  SPLIT: {split}  ({len(label_dirs)} labels)")
        print(f"{'='*62}")

        for label_dir in label_dirs:
            label  = label_dir.name
            videos = sorted([p for p in label_dir.iterdir()
                              if p.suffix.lower() in VIDEO_EXTS])
            n_vids = len(videos)
            if n_vids == 0:
                print(f"\n  [{label}] — không có video")
                continue

            print(f"\n  [{split}/{label}]  {n_vids} videos")

            # Kiểm tra đã gán chưa
            labeled, unlabeled = count_labeled(videos)
            if labeled > 0:
                print(f"    Đã gán: {labeled}/{n_vids}  |  Chưa gán: {unlabeled}/{n_vids}")

            # Skip toàn folder nếu tất cả đã có emotion
            if skip_labeled and unlabeled == 0:
                existing = get_existing_emotion(str(videos[0]))
                mapped   = EMOTION_MAP.get(existing, existing)
                print(f"    ⏭️  Skip (tất cả đã gán: '{existing}' → '{mapped}')")
                summary.append({
                    'split': split, 'label': label, 'n_videos': n_vids,
                    'cnn': existing, 'mapped': mapped,
                    'confidence': -1, 'n_new': 0, 'status': 'skipped',
                })
                continue

            # Chỉ predict video chưa gán (nếu skip_labeled)
            if skip_labeled and unlabeled < n_vids:
                videos_to_predict = [v for v in videos
                                      if get_existing_emotion(str(v)) is None]
                print(f"    Chỉ predict {len(videos_to_predict)} video chưa gán...")
            else:
                videos_to_predict = videos

            # Predict
            result = predict_folder_videos(videos_to_predict, model, cnn_emotions, n_frames)
            if result is None:
                print(f"    ❌ Không đọc được video nào!")
                continue

            emotion_cnn    = result['emotion_cnn']
            emotion_mapped = result['emotion_mapped']
            confidence     = result['confidence']

            sorted_probs = sorted(result['avg_probs'].items(),
                                   key=lambda x: x[1], reverse=True)
            top3_str = "  ".join([f"{e}:{p*100:.1f}%" for e, p in sorted_probs[:3]])
            print(f"    Avg probs (top3) : {top3_str}")
            print(f"    ➜  CNN: '{emotion_cnn}'  →  mapped: '{emotion_mapped}'  "
                  f"(conf: {confidence*100:.1f}%)")
            print(f"    Frames used: {result['n_frames_used']} "
                  f"({n_frames} frames × {result['n_videos']} videos)")

            # ── Xử lý confidence thấp ─────────────────────────────
            is_manual = False
            if confidence < conf_thresh:
                if dry_run:
                    # Trong dry-run chỉ báo, không hỏi
                    print(f"    ⚠️  [DRY RUN] Confidence thấp "
                          f"({confidence*100:.1f}% < {conf_thresh*100:.0f}%)"
                          f" — sẽ hỏi khi chạy thật")
                elif interactive:
                    chosen = prompt_low_conf(split, label, emotion_mapped,
                                             confidence, top3_str, conf_thresh)
                    if chosen is None:
                        # User chọn bỏ qua
                        summary.append({
                            'split': split, 'label': label, 'n_videos': n_vids,
                            'n_new': 0, 'cnn': emotion_cnn,
                            'mapped': emotion_mapped,
                            'confidence': confidence, 'status': 'skipped_low_conf',
                        })
                        continue
                    # Nếu user chọn gán tay (khác với CNN)
                    is_manual = (chosen != emotion_mapped)
                    emotion_mapped = chosen
                    result['emotion_cnn'] = emotion_cnn   # giữ gốc
                    result['emotion_mapped'] = emotion_mapped
                else:
                    # --no-interactive: tự động bỏ qua
                    print(f"    ⏭️  Bỏ qua (conf thấp, --no-interactive)")
                    summary.append({
                        'split': split, 'label': label, 'n_videos': n_vids,
                        'n_new': 0, 'cnn': emotion_cnn,
                        'mapped': emotion_mapped,
                        'confidence': confidence, 'status': 'skipped_low_conf',
                    })
                    continue

            n_will = len(videos_to_predict)
            n_skip = n_vids - n_will

            if dry_run:
                print(f"    [DRY RUN] Sẽ gán '{emotion_mapped}' cho {n_will} video"
                      + (f"  (bỏ qua {n_skip} đã có)" if n_skip > 0 else ""))
            else:
                for vpath in videos_to_predict:
                    save_emotion_json(str(vpath), emotion_mapped, result, manual=is_manual)
                source_tag = "✏️ (gán tay)" if is_manual else "✅"
                skip_note  = f"  (⏭️ skip {n_skip} đã có)" if n_skip > 0 else ""
                print(f"    {source_tag} Đã gán '{emotion_mapped}' → {n_will} file .json{skip_note}")

            summary.append({
                'split':      split,
                'label':      label,
                'n_videos':   n_vids,
                'n_new':      n_will,
                'cnn':        emotion_cnn,
                'mapped':     emotion_mapped,
                'confidence': confidence,
                'status':     'manual' if is_manual else 'processed',
            })

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {'TỔNG KẾT':^58}")
    print(f"{'='*62}")
    print(f"  {'Split':<8} {'Label':<20} {'Total':>6} {'Mới':>5} "
          f"{'Skip':>5}  {'→ Mapped':<15} {'Conf':>7}")
    print(f"  {'-'*65}")

    for r in summary:
        conf_str = f"{r['confidence']*100:>6.1f}%" if r['confidence'] >= 0 else "    — "
        n_new    = r.get('n_new', 0)
        n_skip   = r['n_videos'] - n_new

        if r.get('status') == 'skipped':
            status = "⏭️ "
        elif r.get('status') == 'skipped_low_conf':
            status = "⚠️ "
        elif r.get('status') == 'manual':
            status = "✏️ "
        else:
            status = "✅ "

        print(f"  {r['split']:<8} {r['label']:<20} {r['n_videos']:>6} {n_new:>5} "
              f"{n_skip:>5}  {status}{r['mapped']:<13} {conf_str}")

    total_vids = sum(r['n_videos'] for r in summary)
    total_new  = sum(r.get('n_new', 0) for r in summary)
    total_skip = total_vids - total_new
    print(f"  {'-'*65}")
    print(f"  {len(summary)} folders  |  {total_vids} videos  |  "
          f"Gán mới: {total_new}  |  Skip: {total_skip}")

    # Legend
    print(f"\n  Legend: ✅ auto  ✏️ manual  ⚠️ bỏ qua (conf thấp)  ⏭️ skip (đã có)")

    if dry_run:
        print(f"\n  ⚠️  DRY RUN — chưa gán gì cả!")
        print(f"  Chạy lại không có --dry-run để thực sự gán.")
    else:
        print(f"\n  ✅ Hoàn thành! Tiếp theo chạy:")
        print(f"     python src/video_to_npy.py")
    print(f"{'='*62}\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tự động gán emotion cho video bằng EfficientNet-B2",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python auto_label_emotion.py                          # tất cả (tự skip đã gán)
  python auto_label_emotion.py --split train            # chỉ train
  python auto_label_emotion.py --split train val        # train + val
  python auto_label_emotion.py --label ai lo_so         # chỉ 2 labels
  python auto_label_emotion.py --dry-run                # preview không gán
  python auto_label_emotion.py --frames 12              # 12 frames/video
  python auto_label_emotion.py --no-skip                # ghi đè kể cả đã có
  python auto_label_emotion.py --conf-thresh 0.8        # hỏi khi conf < 80%
  python auto_label_emotion.py --no-interactive         # tự động bỏ conf thấp, không hỏi

  # Gỡ nhãn confidence thấp:
  python auto_label_emotion.py --unlabel-low-conf                   # gỡ conf < 70%
  python auto_label_emotion.py --unlabel-low-conf --conf-thresh 0.8 # gỡ conf < 80%
  python auto_label_emotion.py --unlabel-low-conf --dry-run         # preview trước
        """
    )
    parser.add_argument('--model',     default=MODEL_PATH,
                        help=f'Path model .pth (default: {MODEL_PATH})')
    parser.add_argument('--video_dir', default=VIDEO_DIR,
                        help=f'Thư mục video (default: {VIDEO_DIR})')
    parser.add_argument('--split',     nargs='+', default=SPLITS,
                        choices=SPLITS,
                        help='Splits cần xử lý (default: train val test)')
    parser.add_argument('--label',     nargs='+', default=None,
                        help='Chỉ xử lý các label này')
    parser.add_argument('--frames',    type=int, default=8,
                        help='Số frame sample mỗi video (default: 8)')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Preview kết quả, không gán emotion')
    parser.add_argument('--no-skip',   action='store_true',
                        help='Ghi đè tất cả, kể cả video đã có emotion')
    parser.add_argument('--conf-thresh', type=float, default=CONF_THRESH_DEF,
                        metavar='THRESH',
                        help=f'Ngưỡng confidence (0–1, default: {CONF_THRESH_DEF}). '
                             f'Dưới ngưỡng → hỏi user (hoặc bỏ nếu --no-interactive)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Không hỏi khi conf thấp — tự động bỏ qua những folder đó')
    # ── Unlabel mode ──────────────────────────────────────────────
    parser.add_argument('--unlabel-low-conf', action='store_true',
                        help='Gỡ nhãn tất cả file đã gán có confidence < --conf-thresh.\n'
                             'Kết hợp với --dry-run để preview trước khi xóa.')

    args = parser.parse_args()

    print("\n" + "="*62)
    print("  AUTO LABEL EMOTION — EfficientNet-B2".center(62))
    print("="*62)
    print(f"  Model       : {args.model}")
    print(f"  Video dir   : {args.video_dir}")
    print(f"  Splits      : {args.split}")
    print(f"  Labels      : {args.label or 'all'}")
    print(f"  Conf thresh : {args.conf_thresh*100:.0f}%")

    if args.unlabel_low_conf:
        # ── CHẾ ĐỘ GỠ NHÃN ───────────────────────────────────────
        print(f"  Mode        : UNLABEL LOW-CONF")
        print(f"  Dry run     : {args.dry_run}")
        print("="*62)

        unlabel_low_conf(
            video_dir    = args.video_dir,
            splits       = args.split,
            label_filter = args.label,
            conf_thresh  = args.conf_thresh,
            dry_run      = args.dry_run,
        )
    else:
        # ── CHẾ ĐỘ GÁN NHÃN ──────────────────────────────────────
        print(f"  Frames/v    : {args.frames}")
        print(f"  Dry run     : {args.dry_run}")
        print(f"  Skip labeled: {not args.no_skip}"
              + ("" if not args.no_skip else "  ← --no-skip: sẽ ghi đè tất cả"))
        print(f"  Interactive : {not args.no_interactive}"
              + ("  ← sẽ hỏi khi conf thấp" if not args.no_interactive
                 else "  ← --no-interactive: tự bỏ qua conf thấp"))
        print(f"\n  Emotion mapping (CNN → video_to_npy):")
        for k, v in EMOTION_MAP.items():
            tag = " (same)" if k == v else f" → {v}"
            print(f"    {k:<12}{tag}")

        model, cnn_emotions = load_model(args.model)

        process_all(
            video_dir    = args.video_dir,
            splits       = args.split,
            label_filter = args.label,
            n_frames     = args.frames,
            dry_run      = args.dry_run,
            skip_labeled = not args.no_skip,
            conf_thresh  = args.conf_thresh,
            interactive  = not args.no_interactive,
            model        = model,
            cnn_emotions = cnn_emotions,
        )


if __name__ == '__main__':
    main()