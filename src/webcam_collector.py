"""
<<<<<<< HEAD
auto_cut_preview.py - VSL Video Cutter với folder browser + label manager
=======
webcam_collector.py - Thu thập video ngôn ngữ ký hiệu VSL
==========================================================
Cách chạy:
    python webcam_collector.py

Menu:
    1. Xem thống kê video
    2. Tạo nhãn mới và thu video
    3. Tiếp tục thu video cho nhãn có sẵn
    4. Thu video IDLE (nghỉ / không ký hiệu)
    5. Upload video có sẵn lên HuggingFace
    6. Gán emotion cho video có sẵn
    7. Gộp video thành ZIP và upload lên HuggingFace   ← MỚI
    8. Lưu và thoát

Thay đổi v2:
    - Bỏ qua frame khi thiếu bộ phận trong lúc recording (không ghi frame đó)
    - Dừng ngay lập tức khi thiếu bộ phận quá MISSING_FRAMES_TO_STOP frames liên tiếp
    - Cho phép setup thời gian quay tối đa (max_duration_secs), tự dừng sau khi hết

Thay đổi v3 (fixed):
    - Sửa lỗi manual mode không cập nhật counter
    - Sửa lỗi handedness index có thể gây crash
    - Thêm logging cho exception trong MediaPipe
    - Tách logic recording ra khỏi auto_mode check

Thay đổi v4 (emotion + upload):
    - Thêm tính năng nhập emotion khi tạo nhãn mới hoặc tiếp tục nhãn cũ
    - Emotion được lưu vào file .json cùng với video (tương thích video_to_npy.py)
    - Upload file MP4 được cải tiến: chọn từng file hoặc nhóm, lọc theo nhãn,
      xem preview danh sách trước khi upload

Thay đổi v5 (zip + upload zip):
    - Thêm hàm zip_label_videos(): gộp MP4 của 1 nhãn thành file ZIP
    - Thêm hàm upload_zip_to_hf(): đẩy file ZIP lên HuggingFace dataset
    - Thêm menu mục 7: gộp & upload ZIP, hỗ trợ chọn 1 nhãn / tất cả nhãn,
      chọn split, xem preview trước khi upload

Thay đổi v6 (shared train dir):
    - Chức năng 2 (thu video mới) và 6 (gán emotion) cùng dùng chung
      thư mục TRAIN_DIR = data/videos/train
    - Video thu mới được lưu vào data/videos/train/{label_name}/
    - Gán emotion cũng quét từ data/videos/train/
>>>>>>> 0324310ce4873800e88571022b2b8c86a776acbb
"""

import os, sys, cv2, json, time, argparse, threading, urllib.request, random
import numpy as np
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP = True
except ImportError:
    HAS_MP = False

PREVIEW_W   = 720
PREVIEW_H   = 405
SPLITS      = ["train", "val", "test"]
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm",
               ".MP4", ".MOV", ".AVI", ".MKV"}

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"),
}

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR.parent
                 if _SCRIPT_DIR.name.lower() == "src"
                 else _SCRIPT_DIR)
_ROOT = _SCRIPT_DIR

BG    = "#0d0d14"
BG2   = "#161622"
PANEL = "#1e1e30"
CARD  = "#252538"
ACC   = "#7c3aed"
ACC2  = "#a78bfa"
GRN   = "#10b981"
GRN2  = "#6ee7b7"
RED   = "#ef4444"
RED2  = "#fca5a5"
YEL   = "#f59e0b"
GRAY  = "#6b7280"
GR2   = "#9ca3af"
WHT   = "#f3f4f6"
MONO  = "Courier New"


def ensure_model(name):
    p = _ROOT / name
    if not p.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], str(p))
    return str(p)


def get_existing_labels(data_dir):
    labels = set()
    data_dir = Path(data_dir)
    for split in SPLITS:
        sd = data_dir / split
        if sd.is_dir():
            for d in sd.iterdir():
                if d.is_dir():
                    labels.add(d.name)
    return sorted(labels)


def count_label_clips(data_dir, label):
    counts = {}
    for split in SPLITS:
        d = Path(data_dir) / split / label
        counts[split] = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
    return counts


def next_clip_index(data_dir, label):
    max_idx = 0
    for split in SPLITS:
        d = Path(data_dir) / split / label
        if not d.is_dir():
            continue
        for f in d.glob(f"{label}_*.mp4"):
            try:
                idx = int(f.stem.split("_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


<<<<<<< HEAD
class Detector:
    def __init__(self, cb=None):
        if cb:
            cb("Khoi tao MediaPipe...")
        self.pose = mp_vision.PoseLandmarker.create_from_options(
=======
# ══════════════════════════════════════════════════════════
# ZIP HELPERS  (v5 - MỚI)
# ══════════════════════════════════════════════════════════

def zip_label_videos(label_dir: str, label_name: str,
                     zip_dir: str = None,
                     include_json: bool = True) -> str | None:
    """
    Gộp tất cả file MP4 (và .json emotion tương ứng) của 1 nhãn thành 1 file ZIP.
    """
    mp4_files = sorted([
        f for f in os.listdir(label_dir) if f.endswith('.mp4')
    ])
    if not mp4_files:
        print(f"  [ZIP] Khong co file MP4 nao trong '{label_dir}'")
        return None

    if zip_dir is None:
        zip_dir = os.path.dirname(label_dir)
    os.makedirs(zip_dir, exist_ok=True)

    existing_zips = [
        f for f in os.listdir(zip_dir)
        if f.startswith(f"{label_name}_") and f.endswith('.zip')
    ]
    if existing_zips:
        existing_zips.sort()
        print(f"  [ZIP] Da ton tai {len(existing_zips)} file ZIP cho '{label_name}':")
        for z in existing_zips:
            sz = os.path.getsize(os.path.join(zip_dir, z)) / (1024*1024)
            print(f"    - {z}  ({sz:.1f} MB)")
        print(f"  [ZIP] Bo qua, khong tao lai.")
        return os.path.join(zip_dir, existing_zips[-1])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name  = f"{label_name}_{timestamp}.zip"
    zip_path  = os.path.join(zip_dir, zip_name)

    print(f"\n  [ZIP] Dang gop {len(mp4_files)} file MP4 → '{zip_name}' ...")

    added_files   = 0
    added_jsons   = 0
    total_size_mb = 0.0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in mp4_files:
            fpath = os.path.join(label_dir, fname)
            arcname = os.path.join(label_name, fname)
            zf.write(fpath, arcname)
            total_size_mb += os.path.getsize(fpath) / (1024 * 1024)
            added_files += 1

            if include_json:
                json_path = os.path.splitext(fpath)[0] + ".json"
                if os.path.exists(json_path):
                    zf.write(json_path, os.path.join(label_name,
                             os.path.basename(json_path)))
                    added_jsons += 1

        manifest = {
            'label':        label_name,
            'num_videos':   added_files,
            'num_emotions': added_jsons,
            'created_at':   datetime.now().isoformat(),
            'files':        mp4_files,
        }
        zf.writestr(
            os.path.join(label_name, '_manifest.json'),
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [ZIP] Xong! {added_files} MP4 + {added_jsons} JSON")
    print(f"  [ZIP] Kich thuoc goc: {total_size_mb:.1f} MB  →  ZIP: {zip_size_mb:.1f} MB")
    print(f"  [ZIP] Da luu: {zip_path}")
    return zip_path


def upload_zip_to_hf(zip_path: str, label_name: str,
                     split: str = "train",
                     repo_id: str = None) -> bool:
    """Upload 1 file ZIP lên HuggingFace dataset."""
    if not os.path.exists(zip_path):
        print(f"  [HF-ZIP] File khong ton tai: {zip_path}")
        return False

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n  [HF-ZIP] Chuan bi upload: {os.path.basename(zip_path)}"
          f"  ({zip_size_mb:.1f} MB)  →  split={split}")

    try:
        from huggingface_hub import HfApi
        import collector.hf_upload as hf_mod

        _repo_id = repo_id or getattr(hf_mod, 'HF_REPO_ID', None)
        if not _repo_id:
            print("  [HF-ZIP] Chua cau hinh HF_REPO_ID!")
            return False

        api = HfApi()
        path_in_repo = (f"data/{split}/zips/{label_name}/"
                        f"{os.path.basename(zip_path)}")

        print(f"  [HF-ZIP] Dang upload len: {_repo_id}/{path_in_repo}")
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=path_in_repo,
            repo_id=_repo_id,
            repo_type="dataset",
            commit_message=f"Add zip: {label_name}/{os.path.basename(zip_path)}",
        )
        print(f"  [HF-ZIP] Upload thanh cong!")
        return True

    except ImportError:
        print("  [HF-ZIP] Dung fallback queue-based upload ...")
        result = upload_to_hf(zip_path, label_name, split=split)
        if result:
            _do_flush(f"Add zip {label_name}/{os.path.basename(zip_path)}")
        return bool(result)

    except Exception as e:
        print(f"  [HF-ZIP] Loi upload: {e}")
        return False


# ══════════════════════════════════════════════════════════
# LỚP CHÍNH
# ══════════════════════════════════════════════════════════

class WebcamVideoCollector:

    COUNTDOWN_SECS          = 1.5
    RELAXED_FRAMES_TO_STOP  = 15
    COOLDOWN_SECS           = 2.0
    MISSING_FRAMES_TO_STOP  = 5
    DEFAULT_MAX_DURATION    = 10

    def __init__(self, output_dir='data/videos'):
        self.output_dir    = output_dir
        # ── v6: Thư mục chung cho chức năng 2 & 6 ──────────────────
        # Chức năng 2 (thu video mới) lưu vào đây
        # Chức năng 6 (gán emotion) quét từ đây
        self.train_dir     = os.path.join(output_dir, 'train')
        os.makedirs(self.train_dir, exist_ok=True)
        # ────────────────────────────────────────────────────────────
        self.metadata_path = os.path.join(output_dir, 'metadata.json')
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = self._load_meta()

        init_hf()

        print("\n" + "="*60)
        print(" KHOI TAO MEDIAPIPE DETECTORS ".center(60))
        print("="*60)

        hand_m = download_model('hand_landmarker.task')
        pose_m = download_model('pose_landmarker_heavy.task')
        face_m = download_model('face_landmarker.task')

        self._latest = dict(pose=None, face=None, hands=None, blendshapes=None)
        self._ts = 0

        print("  Khoi tao PoseLandmarker ...")
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
>>>>>>> 0324310ce4873800e88571022b2b8c86a776acbb
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model("pose_landmarker_heavy.task")),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4))
        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model("hand_landmarker.task")),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
<<<<<<< HEAD
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4))

    def detect(self, bgr):
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pr     = self.pose.detect(mp_img)
        hr     = self.hand.detect(mp_img)
        pose   = pr.pose_landmarks[0] if pr.pose_landmarks else None
        hands  = hr.hand_landmarks    if hr.hand_landmarks else []
        return pose, hands
=======
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_hand))

        print("  Khoi tao FaceLandmarker (+ Blendshapes) ...")
        self.face_detector = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                result_callback=self._on_face))

        print("  Tat ca detector da san sang!\n")

    # ── MediaPipe callbacks ───────────────────────────────

    def _on_pose(self, result, image, ts):
        self._latest['pose'] = (result.pose_landmarks[0]
                                if result.pose_landmarks else None)

    def _on_hand(self, result, image, ts):
        left = right = None
        if result.hand_landmarks and result.handedness:
            for i, hlms in enumerate(result.hand_landmarks):
                if i < len(result.handedness) and len(result.handedness[i]) > 0:
                    cat = result.handedness[i][0].category_name
                    if cat == 'Left':
                        right = hlms
                    else:
                        left = hlms
        self._latest['hands'] = (left, right)

    def _on_face(self, result, image, ts):
        self._latest['face'] = (result.face_landmarks[0]
                                if result.face_landmarks else None)
        self._latest['blendshapes'] = (result.face_blendshapes[0]
                                       if result.face_blendshapes else None)

    # ── Metadata ─────────────────────────────────────────

    def _load_meta(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return dict(labels={}, total_videos=0,
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def _save_meta(self):
        self.metadata['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _dn_path(self):
        return os.path.normpath(
            os.path.join(self.output_dir, '..', 'processed', 'display_names.json'))

    def _save_display_name(self, label_key, viet_name):
        path = self._dn_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dn = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                dn = json.load(f)
        if label_key not in dn:
            dn[label_key] = viet_name
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dn, f, indent=2, ensure_ascii=False)
            print(f"  Da luu: '{label_key}' → '{viet_name}'")

    def show_statistics(self):
        print("\n" + "="*60)
        print(" THONG KE VIDEO DA THU ".center(60))
        print("="*60)
        if not self.metadata['labels']:
            print("\n  Chua co video nao")
        else:
            total = 0
            print(f"\n  {'Nhan':<30} {'So video':<15} {'Emotion mac dinh'}")
            print("  " + "-"*65)
            for lb, info in sorted(self.metadata['labels'].items()):
                n   = info.get('num_videos', 0)
                emo = info.get('default_emotion', '-')
                print(f"  {lb:<30} {n:<15} {emo}")
                total += n
            print("  " + "-"*65)
            print(f"  {'TONG CONG':<30} {total}")
        print(f"\n  Cap nhat: {self.metadata.get('updated_at','N/A')}")
        print("="*60)

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _hands_relaxed(pose_lms, left_hand_lms, right_hand_lms) -> bool:
        if left_hand_lms is None and right_hand_lms is None:
            return True
        if pose_lms is None:
            return False
        hip_y = None
        if pose_lms[23].visibility > 0.4 and pose_lms[24].visibility > 0.4:
            hip_y = (pose_lms[23].y + pose_lms[24].y) / 2
        elif pose_lms[23].visibility > 0.4:
            hip_y = pose_lms[23].y
        elif pose_lms[24].visibility > 0.4:
            hip_y = pose_lms[24].y
        elif pose_lms[11].visibility > 0.4 and pose_lms[12].visibility > 0.4:
            hip_y = (pose_lms[11].y + pose_lms[12].y) / 2 + 0.25
        else:
            return False
        margin   = 0.03
        left_ok  = True
        right_ok = True
        if pose_lms[15].visibility > 0.4:
            left_ok  = pose_lms[15].y > (hip_y - margin)
        if pose_lms[16].visibility > 0.4:
            right_ok = pose_lms[16].y > (hip_y - margin)
        return left_ok and right_ok

    @staticmethod
    def _parts_present(pose_lms, left_h, right_h) -> tuple:
        if pose_lms is None:
            return False, "Pose"
        if left_h is None and right_h is None:
            return False, "Ban tay"
        return True, ""

    # ── UI helpers ────────────────────────────────────────

    def _draw_warnings(self, frame, fr, w, h):
        if fr['ok']:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,255,0), 3)
            draw_text_bg(frame, "GOC QUAY: OK", (10, h-60),
                         scale=0.6, color=(0,255,0), bg=(0,50,0))
        else:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 4)
            y = h - 60 - (len(fr['warnings'])-1)*30
            for w_txt in fr['warnings']:
                draw_text_bg(frame, f"! {w_txt}", (10, y),
                             scale=0.55, color=(0,0,255), bg=(50,0,0))
                y += 30
        det = fr['details']
        items = [('Mat', det['face_visible']),
                 ('Than', det['upper_body_visible']),
                 ('Tay T', det['left_arm_visible']),
                 ('Tay P', det['right_arm_visible']),
                 ('Ban tay T', det['left_hand_visible']),
                 ('Ban tay P', det['right_hand_visible'])]
        x0 = w - 130
        for i, (nm, ok) in enumerate(items):
            c = (0,255,0) if ok else (0,0,255)
            cv2.putText(frame, f"{'[OK]' if ok else '[X] '} {nm}",
                        (x0, 70+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    def _draw_expression(self, frame, expr, w, h):
        if expr is None:
            return
        px, py = 10, 70
        src = expr.get('source', '?')
        draw_text_bg(frame,
                     f"Bieu cam: {expr['expression_label']} [{src}]",
                     (px, py), scale=0.55, color=(255,255,0), bg=(40,40,40))
        lines = [
            f"Mieng: {'Mo' if expr['mouth_open']>0.3 else 'Dong'} "
            f"({expr['mouth_open']:.2f})  Cuoi:{expr['mouth_smile']:.2f}",
            f"Mat T:{expr['left_eye_open']:.2f}  Mat P:{expr['right_eye_open']:.2f}"
            f"  Wide:{expr.get('eye_wide',0):.2f}  Squint:{expr.get('eye_squint',0):.2f}",
            f"May len:{expr.get('brow_up',0):.2f}  May xuong:{expr.get('brow_down',0):.2f}",
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (px, py+22+i*17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200,200,200), 1)

    def _draw_interactions(self, frame, interactions, w, h):
        if not interactions:
            return
        y = 195
        draw_text_bg(frame, "TUONG TAC:", (10, y),
                     scale=0.55, color=(0,255,255), bg=(40,40,40))
        for i, txt in enumerate(interactions):
            cv2.putText(frame, f">> {txt}", (10, y+22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

    def _draw_countdown(self, frame, w, h, elapsed_cd, num_color):
        remaining  = max(0, self.COUNTDOWN_SECS - elapsed_cd)
        count_text = str(int(remaining) + 1)
        fs = 4.0
        (ctw, cth), _ = cv2.getTextSize(
            count_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 6)
        cx = (w - ctw) // 2
        cy = (h + cth) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx-40, cy-cth-30),
                      (cx+ctw+40, cy+30), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, count_text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, num_color, 6)
        cv2.putText(frame, "CHUAN BI...", (w//2-80, cy+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        bar_y = cy + 70
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+bar_w, bar_y+12), (80,80,80), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+int(bar_w*(elapsed_cd/self.COUNTDOWN_SECS)), bar_y+12),
                      num_color, -1)

    def _draw_recording(self, frame, w, h, elapsed, frame_count, relaxed_cnt,
                         max_duration=0, missing_cnt=0, skipped=0, is_manual=False,
                         emotion=None):
        if max_duration > 0 and not is_manual:
            remaining  = max(0.0, max_duration - elapsed)
            timer_txt  = f"REC {elapsed:.1f}s / {max_duration}s  |  {frame_count}f"
            timer_color = (0, 0, 255) if remaining > 3 else (0, 80, 255)
        else:
            timer_txt  = f"REC {elapsed:.1f}s | {frame_count}f"
            timer_color = (0, 0, 255)

        if is_manual:
            timer_txt = "[MANUAL] " + timer_txt
            timer_color = (255, 100, 0)

        cv2.putText(frame, timer_txt,
                    (w//2-140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)

        if int(elapsed*2) % 2 == 0:
            cv2.circle(frame, (w//2-165, 20), 8, (0,0,255), -1)

        if emotion:
            emo_color = {
                'happy': (0, 220, 120), 'angry': (0, 60, 255),
                'sad':   (200, 100, 50), 'neutral': (180, 180, 180),
                'surprise': (0, 200, 255), 'fear': (80, 0, 200),
                'disgust': (50, 180, 80),
            }.get(emotion, (200, 200, 200))
            cv2.putText(frame, f"EMO: {emotion.upper()}",
                        (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, emo_color, 2)

        if max_duration > 0 and not is_manual:
            bar_w = int(w * 0.5)
            bar_x = (w - bar_w) // 2
            bar_y = 35
            ratio = min(elapsed / max_duration, 1.0)
            col   = (0,200,0) if ratio < 0.7 else (0,165,255) if ratio < 0.9 else (0,50,255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+6), (50,50,50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*ratio), bar_y+6), col, -1)

        if missing_cnt > 0:
            ratio_m  = missing_cnt / self.MISSING_FRAMES_TO_STOP
            warn_txt = f"! THIEU BO PHAN ({missing_cnt}/{self.MISSING_FRAMES_TO_STOP}f)"
            cv2.putText(frame, warn_txt,
                        (w//2-180, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 255), 2)
            bw2 = 280; bx2 = (w - bw2) // 2
            cv2.rectangle(frame, (bx2, h//2), (bx2+bw2, h//2+10), (60,60,60), -1)
            cv2.rectangle(frame, (bx2, h//2), (bx2+int(bw2*ratio_m), h//2+10),
                          (0, 50, 255), -1)

        if skipped > 0:
            cv2.putText(frame, f"Bo qua: {skipped}f",
                        (10, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,255), 1)

        if relaxed_cnt > 3 and not is_manual:
            ratio = relaxed_cnt / self.RELAXED_FRAMES_TO_STOP
            cv2.putText(frame,
                        f"Tha tay... dung sau {self.RELAXED_FRAMES_TO_STOP - relaxed_cnt}f",
                        (w//2-160, h//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)
            bw3 = 300; bx3 = (w - bw3) // 2
            cv2.rectangle(frame, (bx3, h//2+60), (bx3+bw3, h//2+70), (80,80,80), -1)
            cv2.rectangle(frame, (bx3, h//2+60),
                          (bx3+int(bw3*ratio), h//2+70), (0,200,255), -1)

    # ══════════════════════════════════════════════════════
    # SETUP CONFIG TRƯỚC KHI QUAY
    # ══════════════════════════════════════════════════════

    def _ask_rec_config(self) -> dict:
        print("\n" + "-"*50)
        print(f"  Cau hinh mac dinh:")
        dur_txt = f"{self.DEFAULT_MAX_DURATION}s" if self.DEFAULT_MAX_DURATION > 0 else "khong gioi han"
        print(f"    Thoi gian toi da   : {dur_txt}")
        print(f"    Frame thieu bo phan: {self.MISSING_FRAMES_TO_STOP} frames → dung ngay")
        ans = input("  Tuy chinh? (y/n, mac dinh n): ").strip().lower()
        if ans != 'y':
            return {
                'max_duration':  self.DEFAULT_MAX_DURATION,
                'missing_limit': self.MISSING_FRAMES_TO_STOP,
            }

        try:
            val = input(f"  Thoi gian toi da (giay, 0=khong gioi han) [{self.DEFAULT_MAX_DURATION}]: ").strip()
            max_dur = int(val) if val else self.DEFAULT_MAX_DURATION
            max_dur = max(0, max_dur)
        except ValueError:
            max_dur = self.DEFAULT_MAX_DURATION

        try:
            val = input(f"  So frames thieu bo phan → dung ngay [{self.MISSING_FRAMES_TO_STOP}]: ").strip()
            miss = int(val) if val else self.MISSING_FRAMES_TO_STOP
            miss = max(1, miss)
        except ValueError:
            miss = self.MISSING_FRAMES_TO_STOP

        dur_display = f"{max_dur}s" if max_dur > 0 else "khong gioi han"
        print(f"  → max={dur_display}  missing={miss}f")
        return {'max_duration': max_dur, 'missing_limit': miss}

    # ══════════════════════════════════════════════════════
    # THU THẬP VIDEO
    # ══════════════════════════════════════════════════════

    def collect_label(self, label_name: str, rec_config: dict = None,
                      default_emotion: str = None,
                      use_train_dir: bool = False):
        """
        Thu video cho một nhãn.

        Args:
            label_name     : Tên nhãn
            rec_config     : Cấu hình quay (max_duration, missing_limit)
            default_emotion: Emotion mặc định cho tất cả video
            use_train_dir  : Nếu True → lưu vào self.train_dir/{label_name}/
                             Nếu False → lưu vào self.output_dir/{label_name}/  (hành vi cũ)
        """
        if rec_config is None:
            rec_config = {
                'max_duration':  self.DEFAULT_MAX_DURATION,
                'missing_limit': self.MISSING_FRAMES_TO_STOP,
            }
        max_duration  = rec_config.get('max_duration',  self.DEFAULT_MAX_DURATION)
        missing_limit = rec_config.get('missing_limit', self.MISSING_FRAMES_TO_STOP)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Khong the mo webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── v6: chọn thư mục lưu video ──────────────────
        base_dir  = self.train_dir if use_train_dir else self.output_dir
        label_dir = os.path.join(base_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)
        # ────────────────────────────────────────────────

        video_count = len([f for f in os.listdir(label_dir)
                           if f.endswith('.mp4')])

        state               = 'idle'
        video_writer        = None
        frame_count         = 0
        skipped_frames      = 0
        missing_cnt         = 0
        start_time          = 0
        countdown_start     = 0
        relaxed_cnt         = 0
        last_stop_time      = 0
        fp                  = None
        show_mesh           = True
        auto_mode           = True
        is_manual_recording = False
        current_emotion     = default_emotion

        emo_display = default_emotion if default_emotion else "chua chon"
        dur_display = f"{max_duration}s" if max_duration > 0 else "inf"
        print(f"\n  Nhan: {label_name.upper()} | Da co: {video_count} video")
        print(f"  Luu vao: {label_dir}")
        print(f"  Emotion mac dinh: {emo_display}")
        print(f"  Max: {dur_display} | Missing: {missing_limit}f → dung ngay")
        print("  [SPACE] Thu cong  [A] Auto  [M] Mesh  [E] Doi emotion  [Q] Thoat\n")

        self._ts = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame       = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            h, w        = frame.shape[:2]
            now         = time.time()

            self._ts += 33
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for det in [self.pose_detector, self.hand_detector, self.face_detector]:
                try:
                    det.detect_async(mp_img, self._ts)
                except Exception as e:
                    if hasattr(self, '_last_error_log') and self._last_error_log == str(e):
                        pass
                    else:
                        print(f"  [WARN] MediaPipe error: {e}")
                        self._last_error_log = str(e)

            pose_lms         = self._latest['pose']
            face_lms         = self._latest['face']
            blends           = self._latest['blendshapes']
            left_h, right_h  = self._latest['hands'] or (None, None)

            parts_ok, missing_name = self._parts_present(pose_lms, left_h, right_h)

            FullBodyDrawer.draw_pose(frame, pose_lms, w, h)
            if show_mesh:
                FullBodyDrawer.draw_face_mesh(frame, face_lms, w, h)
            FullBodyDrawer.draw_hand(frame, left_h,  w, h, 'L')
            FullBodyDrawer.draw_hand(frame, right_h, w, h, 'R')

            framing = FramingChecker.check(
                pose_lms, face_lms, (left_h, right_h), w, h)
            self._draw_warnings(frame, framing, w, h)

            expr = (FacialExpressionAnalyzer.analyze_blendshapes(blends)
                    if blends else
                    FacialExpressionAnalyzer.analyze_landmarks(face_lms, w, h))
            self._draw_expression(frame, expr, w, h)

            frame, interactions = InteractionVisualizer.draw(
                frame, pose_lms, face_lms, left_h, right_h, w, h)
            self._draw_interactions(frame, interactions, w, h)

            relaxed = self._hands_relaxed(pose_lms, left_h, right_h)

            if auto_mode:
                if state == 'idle':
                    in_cooldown = (now - last_stop_time) < self.COOLDOWN_SECS
                    if framing['ok'] and not relaxed and not in_cooldown:
                        state = 'countdown'
                        countdown_start = now
                        relaxed_cnt = 0
                        print(f"  Auto: OK → dem nguoc {self.COUNTDOWN_SECS}s...")

                elif state == 'countdown':
                    elapsed_cd = now - countdown_start
                    if elapsed_cd >= self.COUNTDOWN_SECS:
                        state, fp, video_writer, frame_count, start_time = \
                            self._start_recording(label_name, label_dir,
                                                  video_count, fps, width, height, now)
                        skipped_frames      = 0
                        missing_cnt         = 0
                        is_manual_recording = False
                        current_emotion     = default_emotion
                        print(f"  Auto: BAT DAU video {video_count+1}  [emotion={current_emotion or 'none'}]")
                    elif not framing['ok'] or relaxed:
                        state = 'idle'
                        print("  Auto: Huy dem nguoc")

            if state == 'recording':
                elapsed = now - start_time

                if not parts_ok:
                    missing_cnt    += 1
                    skipped_frames += 1
                    if missing_cnt >= missing_limit:
                        print(f"  DUNG NGAY: thieu {missing_name} ({missing_cnt}f lien tiep)")
                        video_count, last_stop_time = self._stop_recording(
                            video_writer, video_count, frame_count,
                            elapsed, fp, label_name, now, current_emotion)
                        video_writer        = None
                        state               = 'idle'
                        missing_cnt         = 0
                        relaxed_cnt         = 0
                        is_manual_recording = False
                else:
                    missing_cnt = 0

                    if auto_mode and not is_manual_recording:
                        relaxed_cnt = relaxed_cnt + 1 if relaxed else 0
                        if relaxed_cnt >= self.RELAXED_FRAMES_TO_STOP:
                            video_count, last_stop_time = self._stop_recording(
                                video_writer, video_count, frame_count,
                                elapsed, fp, label_name, now, current_emotion)
                            video_writer        = None
                            state               = 'idle'
                            relaxed_cnt         = 0
                            missing_cnt         = 0
                        elif max_duration > 0 and elapsed >= max_duration:
                            print(f"  AUTO DUNG: het {max_duration}s")
                            video_count, last_stop_time = self._stop_recording(
                                video_writer, video_count, frame_count,
                                elapsed, fp, label_name, now, current_emotion)
                            video_writer        = None
                            state               = 'idle'
                            relaxed_cnt         = 0
                            missing_cnt         = 0

            if state == 'recording' and video_writer and parts_ok:
                video_writer.write(clean_frame)
                frame_count += 1

            cv2.rectangle(frame, (0,0), (w,55), (30,30,30), -1)
            cv2.putText(frame, f"Nhan: {label_name.upper()}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Video: {video_count}", (10,48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            emo_hdr = default_emotion if default_emotion else "none"
            cv2.putText(frame, f"Emo:{emo_hdr}",
                        (w-330, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,220,255), 1)

            dur_hdr = f"{max_duration}s" if max_duration > 0 else "inf"
            cv2.putText(frame, f"Max:{dur_hdr} Miss:{missing_limit}f",
                        (w-230, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            cv2.putText(frame,
                        f"[A] Auto: {'ON' if auto_mode else 'OFF'}",
                        (w-230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0) if auto_mode else (100,100,100), 1)

            if state == 'countdown':
                elapsed_cd = now - countdown_start
                num_color  = ((0,0,255) if elapsed_cd > self.COUNTDOWN_SECS - 2
                              else (0,165,255) if elapsed_cd > self.COUNTDOWN_SECS - 3
                              else (0,255,0))
                self._draw_countdown(frame, w, h, elapsed_cd, num_color)

            elif state == 'recording':
                self._draw_recording(
                    frame, w, h,
                    now - start_time, frame_count, relaxed_cnt,
                    max_duration, missing_cnt, skipped_frames,
                    is_manual=is_manual_recording,
                    emotion=current_emotion)

            elif state == 'idle':
                in_cooldown = (now - last_stop_time) < self.COOLDOWN_SECS
                if in_cooldown and auto_mode:
                    cv2.putText(frame,
                                f"Cho {self.COOLDOWN_SECS-(now-last_stop_time):.1f}s ...",
                                (w//2-60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200,200,200), 1)
                elif auto_mode:
                    hint = "Gio tay len de bat dau" if relaxed else "Dieu chinh khung hinh..."
                    cv2.putText(frame, hint, (w//2-130, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
                else:
                    cv2.putText(frame, "[SPACE] de bat dau quay",
                                (w//2-120, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0,255,0), 1)

            cv2.putText(frame,
                        "Tay: THA LONG" if relaxed else "Tay: GIO LEN",
                        (w-160, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100,100,255) if relaxed else (0,255,100), 1)

            cv2.rectangle(frame, (0,h-30), (w,h), (30,30,30), -1)
            cv2.putText(frame,
                        "[SPACE] Thu cong  |  [A] Auto  |  [M] Mesh  |  [E] Emotion  |  [Q] Thoat",
                        (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

            cv2.imshow('VSL Collector', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if state != 'recording':
                    if default_emotion is None:
                        print("\n  [!] Chua co emotion! Chon emotion cho video nay:")
                        chosen = ask_emotion()
                        current_emotion = chosen
                    else:
                        current_emotion = default_emotion
                    state = 'recording'
                    fp, video_writer, frame_count, start_time = \
                        self._start_recording_manual(
                            label_name, label_dir, video_count, fps, width, height)
                    skipped_frames      = 0
                    missing_cnt         = 0
                    relaxed_cnt         = 0
                    is_manual_recording = True
                    print(f"  Thu cong: BAT DAU video {video_count+1}  [emotion={current_emotion or 'none'}]")
                else:
                    video_count, last_stop_time = self._stop_recording(
                        video_writer, video_count, frame_count,
                        time.time() - start_time, fp, label_name, time.time(),
                        current_emotion)
                    video_writer        = None
                    state               = 'idle'
                    relaxed_cnt         = 0
                    missing_cnt         = 0
                    skipped_frames      = 0
                    is_manual_recording = False

            elif key in (ord('e'), ord('E')):
                if state != 'recording':
                    print("\n  Doi emotion mac dinh:")
                    new_emo = ask_emotion(default=default_emotion)
                    default_emotion = new_emo
                    print(f"  Emotion moi: {default_emotion or 'none'}")
                else:
                    print("  [!] Khong the doi emotion khi dang ghi!")

            elif key in (ord('a'), ord('A')):
                auto_mode = not auto_mode
                if not auto_mode and state == 'countdown':
                    state = 'idle'
                print(f"  Auto: {'ON' if auto_mode else 'OFF'}")

            elif key in (ord('m'), ord('M')):
                show_mesh = not show_mesh

            elif key in (ord('q'), ord('Q')):
                if state == 'recording' and video_writer:
                    video_writer.release()
                    if current_emotion and fp:
                        save_video_emotion(fp, current_emotion)
                    video_count += 1
                break

        cap.release()
        cv2.destroyAllWindows()
        self._ts = 0

        self.metadata['labels'][label_name] = dict(
            num_videos=video_count,
            path=label_dir,
            default_emotion=default_emotion or '')
        self.metadata['total_videos'] = sum(
            v['num_videos'] for v in self.metadata['labels'].values())
        self._save_meta()
        print(f"\n  Hoan thanh: {label_name} - {video_count} video  [emotion={default_emotion or 'none'}]")

    # ── Start / Stop helpers ──────────────────────────────

    def _make_video_path(self, label_name, label_dir, video_count):
        fn = f'{label_name}_{video_count:04d}.mp4'
        return os.path.join(label_dir, fn)

    def _start_recording(self, label_name, label_dir,
                          video_count, fps, width, height, now):
        fp = self._make_video_path(label_name, label_dir, video_count)
        vw = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
        return 'recording', fp, vw, 0, now

    def _start_recording_manual(self, label_name, label_dir,
                                 video_count, fps, width, height):
        fp = self._make_video_path(label_name, label_dir, video_count)
        vw = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
        return fp, vw, 0, time.time()

    def _stop_recording(self, video_writer, video_count,
                         frame_count, duration, fp, label_name, now,
                         emotion=None):
        video_writer.release()
        video_count += 1
        print(f"  DUNG video {video_count} ({frame_count}f, {duration:.1f}s)  [emotion={emotion or 'none'}]")
        if emotion and fp:
            save_video_emotion(fp, emotion)
        upload_to_hf(fp, label_name, split="train")
        _do_flush(f"Add {label_name}/{os.path.basename(fp)}")
        return video_count, now

    # ══════════════════════════════════════════════════════
    # MENU
    # ══════════════════════════════════════════════════════

    def interactive_menu(self):
        while True:
            print("\n" + "="*60)
            print(" VSL COLLECTOR - MediaPipe Tasks API ".center(60, "="))
            print("="*60)
            print("  1. Xem thong ke video")
            print("  2. Tao nhan moi va thu video  [luu → train/]")
            print("  3. Tiep tuc thu video cho nhan co san  [quet tu train/]")
            print("  4. Thu video IDLE (nghi / khong ky hieu)")
            print("  5. Upload video co san len HuggingFace")
            print("  6. Gan emotion cho video co san  [quet tu train/]")
            print("  7. Gop video thanh ZIP va upload len HuggingFace   ← MOI")
            print("  8. Luu va thoat")
            print("="*60)

            ch = input("\n  Chon (1-8): ").strip()

            if ch == "1":
                self.show_statistics()

            elif ch == "2":
                # ── v6: lưu vào train_dir ────────────────────────
                print(f"\n  [!] Video se duoc luu vao: {self.train_dir}")
                lb = input("\n  Ten nhan moi (khong dau): ").strip().lower().replace(" ", "_")
                if not lb:
                    print("  Ten rong!"); continue
                viet = input(f"  Ten tieng Viet cho '{lb}': ").strip() or lb
                self._save_display_name(lb, viet)

                # Kiểm tra nhãn đã tồn tại trong train_dir
                existing_train = os.path.join(self.train_dir, lb)
                if os.path.isdir(existing_train):
                    n_existing = len([f for f in os.listdir(existing_train)
                                      if f.endswith('.mp4')])
                    if n_existing > 0:
                        print(f"  '{lb}' da co {n_existing} video trong train/! Thu them? (y/n): ", end="")
                        if input().strip().lower() != 'y':
                            continue

                print(f"\n  Chon emotion mac dinh cho tat ca video cua nhan '{lb}':")
                default_emo = ask_emotion(default='neutral')

                rec_cfg = self._ask_rec_config()
                self.collect_label(lb, rec_config=rec_cfg,
                                   default_emotion=default_emo,
                                   use_train_dir=True)   # ← lưu vào train/

            elif ch == "3":
                # ── v6: quét nhãn trực tiếp từ train_dir ────────
                print(f"\n  [!] Quet nhan tu: {self.train_dir}")
                if not os.path.isdir(self.train_dir):
                    print("  Thu muc train/ chua ton tai!"); continue

                labels = sorted([
                    d for d in os.listdir(self.train_dir)
                    if os.path.isdir(os.path.join(self.train_dir, d))
                ])
                if not labels:
                    print("  Chua co nhan nao trong train/!"); continue

                print("\n  Danh sach nhan:")
                for i, lb in enumerate(labels, 1):
                    lb_path = os.path.join(self.train_dir, lb)
                    n       = len([f for f in os.listdir(lb_path) if f.endswith('.mp4')])
                    emo     = self.metadata['labels'].get(lb, {}).get('default_emotion', '-')
                    print(f"  {i:>3}. {lb} ({n} video)  [emo: {emo}]")

                try:
                    idx = int(input("\n  Chon so: ").strip()) - 1
                    if 0 <= idx < len(labels):
                        chosen_lb = labels[idx]
                        saved_emo = self.metadata['labels'].get(chosen_lb, {}).get('default_emotion') or None

                        print(f"\n  Emotion hien tai cua '{chosen_lb}': {saved_emo or 'chua co'}")
                        ans = input("  Doi emotion? (y/n, mac dinh n): ").strip().lower()
                        if ans == 'y':
                            new_emo = ask_emotion(default=saved_emo or 'neutral')
                        else:
                            new_emo = saved_emo

                        rec_cfg = self._ask_rec_config()
                        self.collect_label(chosen_lb, rec_config=rec_cfg,
                                           default_emotion=new_emo,
                                           use_train_dir=True)   # ← lưu vào train/
                    else:
                        print("  Khong hop le!")
                except ValueError:
                    print("  Nhap so!")

            elif ch == "4":
                self._menu_idle()

            elif ch == "5":
                self._menu_upload_files()

            elif ch == "6":
                # ── v6: gán emotion quét từ train_dir ───────────
                self._menu_assign_existing_emotions()

            elif ch == "7":
                self._menu_zip_and_upload()

            elif ch == "8":
                self._save_meta()
                self.show_statistics()
                self._ask_organize_on_exit()
                print("\n  Tam biet!\n")
                break
            else:
                print("  Khong hop le!")

    # ══════════════════════════════════════════════════════
    # MENU 7: GỘP VIDEO THÀNH ZIP VÀ UPLOAD  (v5 - MỚI)
    # ══════════════════════════════════════════════════════

    def _menu_zip_and_upload(self):
        print("\n" + "="*60)
        print(" GOP VIDEO THANH ZIP + UPLOAD HUGGINGFACE ".center(60))
        print("="*60)

        available = []
        skip_dirs = {'train', 'val', 'test'}
        for lb in sorted(os.listdir(self.output_dir)):
            lp = os.path.join(self.output_dir, lb)
            if not os.path.isdir(lp) or lb in skip_dirs:
                continue
            mp4s = [f for f in os.listdir(lp) if f.endswith('.mp4')]
            if mp4s:
                available.append((lb, lp, len(mp4s)))

        # Quét thêm từ train_dir
        if os.path.isdir(self.train_dir):
            for lb in sorted(os.listdir(self.train_dir)):
                lp = os.path.join(self.train_dir, lb)
                if not os.path.isdir(lp):
                    continue
                mp4s = [f for f in os.listdir(lp) if f.endswith('.mp4')]
                if mp4s and not any(x[0] == lb for x in available):
                    available.append((lb, lp, len(mp4s)))

        if not available:
            print("\n  Khong co nhan nao co video!")
            input("  ENTER de quay lai..."); return

        print(f"\n  {'#':<5} {'Nhan':<30} {'So MP4':<10} {'Tong kich thuoc'}")
        print("  " + "-"*60)
        for i, (lb, lp, n) in enumerate(available, 1):
            total_mb = sum(
                os.path.getsize(os.path.join(lp, f)) / (1024*1024)
                for f in os.listdir(lp) if f.endswith('.mp4')
            )
            print(f"  {i:<5} {lb:<30} {n:<10} {total_mb:.1f} MB")
        print("  " + "-"*60)
        print("   0. Gop tat ca nhan")
        print()

        raw = input("  Chon nhan (0 / so / nhieu so cach boi dau phay): ").strip()

        if raw == "0":
            chosen = available
        else:
            chosen = []
            for tok in raw.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    idx = int(tok) - 1
                    if 0 <= idx < len(available):
                        chosen.append(available[idx])
                    else:
                        print(f"  Bo qua so ngoai pham vi: {tok}")
                else:
                    match = [(lb, lp, n) for lb, lp, n in available if lb == tok]
                    chosen.extend(match)

        if not chosen:
            print("  Khong chon duoc nhan nao!"); return

        total_videos = sum(n for _, _, n in chosen)
        print(f"\n  Se gop {len(chosen)} nhan, tong {total_videos} video:")
        for lb, _, n in chosen:
            print(f"    - {lb}: {n} video")

        default_zip_dir = os.path.join(os.path.dirname(self.output_dir), 'zips')
        print(f"\n  Thu muc luu ZIP (mac dinh: {default_zip_dir})")
        zip_dir_input = input("  Nhap duong dan hoac Enter de dung mac dinh: ").strip()
        zip_dir = zip_dir_input if zip_dir_input else default_zip_dir
        os.makedirs(zip_dir, exist_ok=True)
        print(f"  → Luu ZIP vao: {zip_dir}")

        inc_json_ans = input("\n  Kem theo file .json emotion? (y/n, mac dinh y): ").strip().lower()
        include_json = (inc_json_ans != 'n')

        print(f"\n  Xac nhan:")
        print(f"    Nhan     : {', '.join(lb for lb, _, _ in chosen)}")
        print(f"    Luu vao  : {zip_dir}")
        print(f"    Kem JSON : {'Co' if include_json else 'Khong'}")
        if input("\n  Bat dau gop? (y/n): ").strip().lower() != 'y':
            print("  Da huy."); return

        skip_labels = []
        for lb, lp, _ in chosen:
            existing = sorted([
                f for f in os.listdir(zip_dir)
                if f.startswith(f"{lb}_") and f.endswith('.zip')
            ])
            if existing:
                sz = os.path.getsize(os.path.join(zip_dir, existing[-1])) / (1024*1024)
                print(f"  [SKIP] '{lb}' da co ZIP: {existing[-1]}  ({sz:.1f} MB)")
                skip_labels.append(lb)

        chosen = [(lb, lp, n) for lb, lp, n in chosen if lb not in skip_labels]

        if not chosen:
            print("\n  Tat ca nhan da co ZIP, khong can tao lai!")
            input("  ENTER de quay lai..."); return

        if skip_labels:
            print(f"  → Se chi gop {len(chosen)} nhan con lai.\n")

        created_zips = []
        for lb, lp, _ in chosen:
            zp = zip_label_videos(lp, lb, zip_dir=zip_dir, include_json=include_json)
            if zp:
                created_zips.append((zp, lb))

        if not created_zips:
            print("\n  Khong tao duoc file ZIP nao!")
            input("  ENTER de quay lai..."); return

        print(f"\n  Da tao {len(created_zips)} file ZIP:")
        for zp, lb in created_zips:
            sz = os.path.getsize(zp) / (1024*1024)
            print(f"    {os.path.basename(zp)}  ({sz:.1f} MB)  ← {lb}")

        if input("\n  Upload ZIP len HuggingFace? (y/n): ").strip().lower() != 'y':
            print(f"\n  Cac file ZIP da duoc luu tai: {zip_dir}")
            input("  ENTER de quay lai..."); return

        print("\n  Upload vao split nao?")
        print("  [1] train  [2] val  [3] test")
        sp_ch = input("  Chon (mac dinh: train): ").strip()
        split = {'1': 'train', '2': 'val', '3': 'test'}.get(sp_ch, 'train')
        print(f"  → split: {split}")

        upload_ok  = []
        upload_fail = []
        for i, (zp, lb) in enumerate(created_zips, 1):
            print(f"\n  [{i}/{len(created_zips)}] Upload '{os.path.basename(zp)}' ...")
            ok = upload_zip_to_hf(zp, lb, split=split)
            if ok:
                upload_ok.append((zp, lb))
            else:
                upload_fail.append((zp, lb))

        print(f"\n  KET QUA UPLOAD:")
        print(f"    Thanh cong : {len(upload_ok)}/{len(created_zips)} file")
        if upload_fail:
            print(f"    That bai   : {len(upload_fail)} file")
            for zp, lb in upload_fail:
                print(f"      - {os.path.basename(zp)}")

        if upload_ok:
            del_ans = input("\n  Xoa file ZIP cuc bo sau khi upload? (y/n, mac dinh n): ").strip().lower()
            if del_ans == 'y':
                for zp, _ in upload_ok:
                    try:
                        os.remove(zp)
                        print(f"  Da xoa: {os.path.basename(zp)}")
                    except Exception as e:
                        print(f"  Loi xoa {os.path.basename(zp)}: {e}")
            else:
                print(f"  Giu lai tai: {zip_dir}")

        input("\n  ENTER de quay lai menu...")

    # ══════════════════════════════════════════════════════
    # MENU 6: GÁN EMOTION CHO VIDEO CÓ SẴN
    # v6: quét từ self.train_dir thay vì self.output_dir
    # ══════════════════════════════════════════════════════

    def _menu_assign_existing_emotions(self):
        print("\n" + "="*60)
        print(" GAN EMOTION CHO VIDEO CO SAN ".center(60))
        print("="*60)
        # ── v6: hiển thị thư mục đang quét ─────────────
        print(f"\n  Dang quet tu: {self.train_dir}")

        label_videos = {}

        # Quét train_dir (chức năng 2 lưu vào đây)
        scan_root = self.train_dir
        if not os.path.isdir(scan_root):
            print(f"\n  Thu muc khong ton tai: {scan_root}")
            input("  ENTER de quay lai..."); return

        for lb_name in sorted(os.listdir(scan_root)):
            lb_path = os.path.join(scan_root, lb_name)
            if not os.path.isdir(lb_path):
                continue
            videos = sorted([f for f in os.listdir(lb_path) if f.endswith('.mp4')])
            if not videos:
                continue
            missing  = []
            assigned = []
            for vf in videos:
                vp  = os.path.join(lb_path, vf)
                emo = get_video_emotion(vp)
                if emo:
                    assigned.append((vp, emo))
                else:
                    missing.append(vp)
            label_videos[lb_name] = {'missing': missing, 'assigned': assigned}

        if not label_videos:
            print(f"\n  Khong tim thay video nao trong '{scan_root}'!")
            input("  ENTER de quay lai..."); return

        total_missing = sum(len(v['missing']) for v in label_videos.values())
        print(f"\n  {'Nhan':<30} {'Co emotion':<15} {'Thieu emotion'}")
        print("  " + "-"*55)
        for lb, data in label_videos.items():
            ok  = len(data['assigned'])
            mis = len(data['missing'])
            status = f"!  {mis}" if mis > 0 else "OK"
            print(f"  {lb:<30} {ok:<15} {status}")
        print("  " + "-"*55)
        print(f"  Tong thieu: {total_missing} video\n")

        if total_missing == 0:
            print("  Tat ca video da co emotion!")
            input("  ENTER de quay lai..."); return

        print("  [1] Gan emotion cho 1 nhan")
        print("  [2] Gan emotion cho tat ca nhan thieu")
        print("  [3] Gan theo tung video (manual)")
        print("  [0] Quay lai")
        sub = input("\n  Chon: ").strip()

        if sub == "0":
            return

        elif sub == "1":
            labels_with_missing = [lb for lb, d in label_videos.items() if d['missing']]
            print("\n  Nhan co video thieu emotion:")
            for i, lb in enumerate(labels_with_missing, 1):
                print(f"  {i:>3}. {lb}  ({len(label_videos[lb]['missing'])} video)")
            try:
                idx = int(input("\n  Chon so: ").strip()) - 1
                if 0 <= idx < len(labels_with_missing):
                    lb   = labels_with_missing[idx]
                    data = label_videos[lb]
                    print(f"\n  Gan emotion cho {len(data['missing'])} video cua '{lb}':")
                    emo = ask_emotion(default='neutral')
                    if emo:
                        for vp in data['missing']:
                            save_video_emotion(vp, emo)
                            print(f"    + {os.path.basename(vp)} → {emo}")
                        print(f"  Xong! {len(data['missing'])} video da duoc gan '{emo}'")
                else:
                    print("  Khong hop le!")
            except ValueError:
                print("  Nhap so!")

        elif sub == "2":
            print(f"\n  Gan emotion cho {total_missing} video thieu:")
            emo = ask_emotion(default='neutral')
            if emo:
                count = 0
                for lb, data in label_videos.items():
                    for vp in data['missing']:
                        save_video_emotion(vp, emo)
                        count += 1
                print(f"  Xong! {count} video da duoc gan '{emo}'")

        elif sub == "3":
            print("\n  Gang theo tung video (s=skip, q=thoat):")
            for lb, data in label_videos.items():
                if not data['missing']:
                    continue
                print(f"\n  ── {lb} ──  ({len(data['missing'])} video thieu)")
                for vp in data['missing']:
                    print(f"    {os.path.basename(vp)}")
                    raw = input("    Emotion (1-7 / ten / s=skip / q=thoat): ").strip().lower()
                    if raw == 'q':
                        return
                    if raw == 's':
                        continue
                    if raw.isdigit() and 1 <= int(raw) <= len(EMOTIONS_LIST):
                        emo = EMOTIONS_LIST[int(raw) - 1]
                    elif raw in EMOTIONS:
                        emo = raw
                    else:
                        print("    Khong hop le, bo qua.")
                        continue
                    save_video_emotion(vp, emo)
                    print(f"    → {emo}")

        input("\n  ENTER de quay lai...")

    # ══════════════════════════════════════════════════════
    # MENU: IDLE
    # ══════════════════════════════════════════════════════

    def _menu_idle(self):
        idle_actions = [
            ("tay_xuoi_hong",     "Tay xuoi ben hong dung yen"),
            ("tay_khoanh_nguc",   "Tay khoanh truoc nguc"),
            ("tay_tren_ban",      "Tay dat tren ban"),
            ("ga_dau",            "Ga dau / chinh toc"),
            ("dua_tay_len_xuong", "Dua tay len roi ha xuong khong ky"),
            ("vuon_vai",          "Vuon vai / doi tu the"),
            ("dung_xa",           "Dung xa camera"),
            ("dung_gan",          "Dung gan camera"),
            ("nghieng_nguoi",     "Nghieng nguoi sang trai phai"),
            ("chi_tro",           "Chi tay ve phia truoc"),
            ("bo_tay_vao_tui",    "Bo tay vao tui quan"),
            ("voi_lay_do",        "Voi tay lay do"),
        ]
        print("\n" + "="*60)
        print(" THU VIDEO IDLE ".center(60))
        print("="*60)
        for i, (key, desc) in enumerate(idle_actions, 1):
            label    = f"__idle__{key}"
            existing = self.metadata['labels'].get(label, {}).get('num_videos', 0)
            status   = f"({existing} video)" if existing else "(chua co)"
            print(f"  {i:>2}. {desc:<40} {status}")
        print("   0. Thu tat ca theo thu tu")
        print("  99. Nhap ten hanh dong rieng")
        print("="*60)
        try:
            choice  = input("\n  Chon (0 / 1-12 / 99): ").strip()
            rec_cfg = self._ask_rec_config()

            print("\n  Emotion cho video IDLE (thuong la neutral):")
            idle_emo = ask_emotion(default='neutral')

            if choice == "0":
                for key, desc in idle_actions:
                    label = f"__idle__{key}"
                    input(f"\n  Chuan bi: {desc}\n  Nhan ENTER de bat dau...")
                    self.collect_label(label, rec_config=rec_cfg, default_emotion=idle_emo)
            elif choice == "99":
                custom = input("  Ten hanh dong (vd: nhin_dien_thoai): ").strip()
                if not custom:
                    print("  Ten rong!"); return
                label = f"__idle__{custom}"
                viet  = input(f"  Ten tieng Viet cho '{label}': ").strip() or label
                self._save_display_name(label, viet)
                self.collect_label(label, rec_config=rec_cfg, default_emotion=idle_emo)
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(idle_actions):
                    key, desc = idle_actions[idx]
                    label = f"__idle__{key}"
                    print(f"\n  Chuan bi: {desc}")
                    self.collect_label(label, rec_config=rec_cfg, default_emotion=idle_emo)
                else:
                    print("  Khong hop le!")
        except ValueError:
            print("  Nhap so!")

    # ══════════════════════════════════════════════════════
    # MENU: UPLOAD FILE MP4
    # ══════════════════════════════════════════════════════

    def _pick_files_gui(self) -> list:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.attributes('-topmost', True)
            paths = filedialog.askopenfilenames(
                title="Chon file video de upload",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                           ("All files", "*.*")])
            root.destroy()
            return list(paths)
        except Exception:
            return []

    def _pick_folder_gui(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.attributes('-topmost', True)
            folder = filedialog.askdirectory(title="Chon thu muc chua video")
            root.destroy()
            return folder
        except Exception:
            return ""

    def _scan_videos_in_path(self, path: str) -> list:
        exts  = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        found = []
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in exts:
                found.append(path)
        elif os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                fp = os.path.join(path, f)
                if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in exts:
                    found.append(fp)
        return found

    def _preview_files(self, files: list):
        print(f"\n  {'#':<5} {'File':<50} {'Emotion'}")
        print("  " + "-"*70)
        for i, fp in enumerate(files, 1):
            emo = get_video_emotion(fp) or "(chua co)"
            fname = os.path.basename(fp)
            if len(fname) > 45:
                fname = "..." + fname[-42:]
            print(f"  {i:<5} {fname:<50} {emo}")
        print("  " + "-"*70)
        print(f"  Tong: {len(files)} file\n")

    def _menu_upload_files(self):
        from collector.hf_upload import init_hf, HF_REPO_ID
        print("\n" + "="*60)
        print(" UPLOAD VIDEO LEN HUGGINGFACE ".center(60))
        print("="*60)
        if init_hf is None:
            print("\n  CANH BAO: HuggingFace chua duoc ket noi!")
            input("\n  Nhan ENTER de quay lai menu..."); return

        print("\n  Chon nguon file:")
        print("  [1] Hop thoai GUI — chon file")
        print("  [2] Hop thoai GUI — chon thu muc")
        print("  [3] Nhap duong dan thu cong")
        print("  [4] Lay tu thu muc output hien tai (data/videos)")
        ch = input("\n  Chon (1-4): ").strip()

        selected_files = []

        if ch == "1":
            selected_files = self._pick_files_gui()
            if not selected_files:
                print("  Khong chon file nao.")

        elif ch == "2":
            folder = self._pick_folder_gui()
            if folder:
                selected_files = self._scan_videos_in_path(folder)
                print(f"  Tim thay {len(selected_files)} file trong '{folder}'")
            else:
                print("  Khong chon thu muc nao.")

        elif ch == "3":
            print("\n  Nhap duong dan file hoac thu muc (dong trong de ket thuc):")
            while True:
                p = input("  > ").strip().strip('"').strip("'")
                if not p:
                    break
                found = self._scan_videos_in_path(p)
                if found:
                    selected_files.extend(found)
                    print(f"    → {len(found)} file them vao danh sach")
                else:
                    print(f"    Khong tim thay: {p}")

        elif ch == "4":
            labels_dirs = []
            for lb in sorted(os.listdir(self.output_dir)):
                lp = os.path.join(self.output_dir, lb)
                if os.path.isdir(lp) and lb not in ('train','val','test'):
                    labels_dirs.append(lb)

            if not labels_dirs:
                print("  Khong co nhan nao trong data/videos!"); return

            print("\n  Nhan co san:")
            for i, lb in enumerate(labels_dirs, 1):
                n = len([f for f in os.listdir(os.path.join(self.output_dir, lb))
                         if f.endswith('.mp4')])
                print(f"  {i:>3}. {lb} ({n} video)")
            print("   0. Chon tat ca nhan")

            raw = input("\n  Chon (0 / so / nhieu so cach nhau boi dau phay): ").strip()
            chosen_labels = []
            if raw == "0":
                chosen_labels = labels_dirs
            else:
                for tok in raw.split(","):
                    tok = tok.strip()
                    try:
                        idx = int(tok) - 1
                        if 0 <= idx < len(labels_dirs):
                            chosen_labels.append(labels_dirs[idx])
                    except ValueError:
                        if tok in labels_dirs:
                            chosen_labels.append(tok)

            for lb in chosen_labels:
                lp = os.path.join(self.output_dir, lb)
                selected_files.extend(self._scan_videos_in_path(lp))
            print(f"  Tong: {len(selected_files)} file tu {len(chosen_labels)} nhan")

        if not selected_files:
            input("  Khong co file. ENTER de quay lai..."); return

        self._preview_files(selected_files)

        files_no_emo = [f for f in selected_files if not get_video_emotion(f)]
        if files_no_emo:
            print(f"  ! {len(files_no_emo)}/{len(selected_files)} file chua co emotion!")
            print("  [1] Gan cung 1 emotion cho tat ca file thieu")
            print("  [2] Bo qua (upload khong co emotion)")
            sub = input("  Chon: ").strip()
            if sub == "1":
                emo = ask_emotion(default='neutral')
                if emo:
                    for fp in files_no_emo:
                        save_video_emotion(fp, emo)
                    print(f"  Da gan '{emo}' cho {len(files_no_emo)} file")

        print(f"\n  Xac nhan upload {len(selected_files)} file?")
        confirm = input("  (y=upload / n=huy / f=loc bot): ").strip().lower()
        if confirm == 'n':
            print("  Da huy."); return
        if confirm == 'f':
            print("\n  Nhap so thu tu file muon XOA khoi danh sach (cach nhau boi phay):")
            self._preview_files(selected_files)
            raw = input("  So can xoa: ").strip()
            to_remove = set()
            for tok in raw.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    idx = int(tok) - 1
                    if 0 <= idx < len(selected_files):
                        to_remove.add(idx)
            selected_files = [f for i, f in enumerate(selected_files) if i not in to_remove]
            print(f"  Con lai {len(selected_files)} file sau khi loc.")
            if not selected_files:
                print("  Danh sach rong, huy upload."); return

        if self.metadata['labels']:
            labels_exist = list(self.metadata['labels'].keys())
            print("\n  Label hien co:")
            for i, lb in enumerate(labels_exist, 1):
                print(f"    {i}. {lb}")
            ans = input("  Nhap so hoac ten label moi: ").strip()
            try:
                idx = int(ans) - 1
                label_name = labels_exist[idx] if 0 <= idx < len(labels_exist) \
                             else ans.lower().replace(" ", "_")
            except ValueError:
                label_name = ans.lower().replace(" ", "_")
        else:
            label_name = input("  Ten label: ").strip().lower().replace(" ", "_")
        if not label_name:
            print("  Ten rong! Huy."); return

        print("\n  Upload vao split nao?")
        print("  [1] train  [2] val  [3] test")
        sp_ch = input("  Chon (mac dinh: train): ").strip()
        split = {'1': 'train', '2': 'val', '3': 'test'}.get(sp_ch, 'train')
        print(f"  → split: {split}")

        label_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        success = 0
        failed  = []
        print(f"\n  Dang queue {len(selected_files)} file...")
        for i, fp in enumerate(selected_files, 1):
            fname        = os.path.basename(fp)
            local_target = os.path.join(label_dir, fname)
            if os.path.abspath(fp) != os.path.abspath(local_target):
                import shutil
                shutil.copy2(fp, local_target)
                src_json = os.path.splitext(fp)[0] + ".json"
                dst_json = os.path.splitext(local_target)[0] + ".json"
                if os.path.exists(src_json):
                    shutil.copy2(src_json, dst_json)
            emo = get_video_emotion(local_target) or "none"
            queued = upload_to_hf(local_target, label_name, split=split)
            status = "queued" if queued else "skip(da upload)"
            print(f"  [{i}/{len(selected_files)}] {fname}  [emo:{emo}]  → {status}")
            if queued:
                success += 1

        if success > 0:
            print(f"\n  Dang commit {success} file len HuggingFace (split={split})...")
            _do_flush(f"Upload {success} videos for {label_name} ({split})")
        else:
            print(f"\n  Khong co file moi can upload (tat ca da upload truoc do).")

        existing = self.metadata['labels'].get(label_name, {}).get('num_videos', 0)
        self.metadata['labels'][label_name] = dict(
            num_videos=existing + success, path=label_dir)
        self.metadata['total_videos'] = sum(
            v['num_videos'] for v in self.metadata['labels'].values())
        self._save_meta()

        print(f"\n  Tong: {success}/{len(selected_files)} file moi da duoc commit  (split={split})")
        skipped_count = len(selected_files) - success
        if skipped_count > 0:
            print(f"  Bo qua: {skipped_count} file (da upload truoc do)")
        input("\n  ENTER de quay lai...")

    # ══════════════════════════════════════════════════════
    # ORGANIZE ON EXIT
    # ══════════════════════════════════════════════════════

    def _ask_organize_on_exit(self):
        video_dir   = self.output_dir
        unorganized = []
        skip        = {'train', 'val', 'test'}
        for entry in os.scandir(video_dir):
            if entry.is_dir() and entry.name not in skip:
                videos = [f for f in os.listdir(entry.path)
                          if f.endswith(('.mp4','.avi','.mov','.mkv','.webm'))]
                if videos:
                    unorganized.append((entry.name, len(videos)))
        if not unorganized:
            return
        print("\n" + "="*60)
        print(" CHIA TRAIN / VAL / TEST ".center(60))
        print("="*60)
        print(f"\n  {len(unorganized)} label chua duoc chia:")
        for lb, n in unorganized:
            print(f"    - {lb}: {n} video")
        if input("\n  Chia ngay bay gio? (y/n): ").strip().lower() != 'y':
            print("  Bo qua."); return
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            from organize_dataset import organize
            stats = organize(src_dir=video_dir, dry_run=False)
            if stats:
                print(f"  Da chia xong {len(stats)} labels!")
        except Exception as e:
            print(f"  Loi: {e}\n  Chay: python organize_dataset.py")
>>>>>>> 0324310ce4873800e88571022b2b8c86a776acbb

    def close(self):
        self.pose.close()
        self.hand.close()


def _shoulder_y(pose):
    if pose is None:
        return None
    ys = [pose[i].y for i in [11, 12] if pose[i].visibility > 0.3]
    return float(np.mean(ys)) if ys else None


def _wrist_min_y(pose, hands):
    ys = []
    for h in hands:
        ys += [h[0].y, h[8].y, h[4].y]
    if not ys and pose:
        ys = [pose[i].y for i in [15, 16, 19, 20]
              if pose[i].visibility > 0.3]
    return min(ys) if ys else None


def _finger_curl_active(hands, curl_thresh=0.15):
    if not hands:
        return False
    FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    for hand in hands:
        curls = []
        for mcp_i, pip_i, tip_i in FINGERS:
            mcp = np.array([hand[mcp_i].x, hand[mcp_i].y])
            pip = np.array([hand[pip_i].x, hand[pip_i].y])
            tip = np.array([hand[tip_i].x, hand[tip_i].y])
            v1 = pip - mcp
            v2 = tip - pip
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                curls.append(0.0)
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curls.append(1.0 - cos_a)
        if float(np.mean(curls)) > curl_thresh:
            return True
    return False


def analyze_frames(frames, fps,
                   padding_sec=0.25, min_dur=0.4, idle_sec=0.7,
                   shoulder_margin=0.03, smooth_w=7,
                   use_margin=True, use_finger=False,
                   finger_curl_thresh=0.15,
                   progress_cb=None):
    det = Detector(progress_cb)
    N   = len(frames)
    raw = np.zeros(N, dtype=np.float32)

    for i, frame in enumerate(frames):
        if progress_cb and i % 15 == 0:
            progress_cb(f"Phan tich frame {i}/{N}  ({i*100//N}%)")
        pose, hands = det.detect(frame)

        cond_margin = True
        cond_finger = True

        if use_margin:
            sy = _shoulder_y(pose)
            wy = _wrist_min_y(pose, hands)
            # wy < sy - shoulder_margin
            # margin am (-0.2) => wy < sy + 0.2 => tay duoc phep thap hon vai 20%
            cond_margin = (sy is not None and wy is not None
                           and wy < sy - shoulder_margin)

        if use_finger:
            cond_finger = _finger_curl_active(hands, finger_curl_thresh)

        if use_margin and use_finger:
            active = cond_margin and cond_finger
        elif use_margin:
            active = cond_margin
        elif use_finger:
            active = cond_finger and bool(hands)
        else:
            active = bool(hands)

        if active:
            raw[i] = 1.0

    det.close()

    k      = np.ones(smooth_w, dtype=np.float32) / smooth_w
    signal = np.convolve(raw, k, mode="same")
    binary = (signal > 0.3).astype(np.int32)

    segs, in_s, s0 = [], False, 0
    for i in range(N):
        if binary[i] and not in_s:
            in_s, s0 = True, i
        elif not binary[i] and in_s:
            in_s = False
            segs.append([s0, i - 1])
    if in_s:
        segs.append([s0, N - 1])
    if not segs:
        return []

    gap    = int(idle_sec * fps)
    merged = [segs[0][:]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_f  = int(min_dur * fps)
    pad    = int(padding_sec * fps)
    result = []
    for s, e in merged:
        if e - s >= min_f:
            result.append([max(0, s - pad), min(N - 1, e + pad)])
    return result


class LabelDialog(tk.Toplevel):
    def __init__(self, parent, data_dir, n_clips):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.n_clips  = n_clips
        self.result   = None

        self.title("Chon nhan & split")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        existing = get_existing_labels(data_dir)
        self._build(existing)
        self.wait_window()

    def _build(self, existing):
        pad = dict(padx=20)

        tk.Label(self, text="Gan nhan & Split cho clip",
                 bg=BG, fg=ACC2,
                 font=(MONO, 13, "bold")).pack(pady=(16, 2), **pad)
        tk.Label(self,
                 text=f"Se luu {self.n_clips} clip duoc chon",
                 bg=BG, fg=GR2,
                 font=(MONO, 9)).pack(pady=(0, 2))
        tk.Label(self,
                 text=f"Luu vao: {self.data_dir}",
                 bg=BG, fg=YEL,
                 font=(MONO, 8)).pack(pady=(0, 8))

        tk.Label(self, text="Nhan co san  (click de chon):",
                 bg=BG, fg=WHT,
                 font=(MONO, 9, "bold")).pack(anchor="w", **pad)

        lf = tk.Frame(self, bg=PANEL, bd=1, relief=tk.SOLID)
        lf.pack(fill=tk.X, padx=20, pady=(4, 0))

        self._lbl_var = tk.StringVar()

        if existing:
            sb = tk.Scrollbar(lf, orient=tk.VERTICAL)
            self._lb = tk.Listbox(
                lf, listvariable=tk.StringVar(value=existing),
                yscrollcommand=sb.set,
                selectmode=tk.SINGLE,
                bg=CARD, fg=WHT, selectbackground=ACC,
                font=(MONO, 10),
                height=min(7, len(existing)),
                relief=tk.FLAT, highlightthickness=0, bd=0)
            sb.config(command=self._lb.yview)
            self._lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            def _sel(evt):
                sel = self._lb.curselection()
                if sel:
                    self._lbl_var.set(existing[sel[0]])
                    self._new_ent.delete(0, tk.END)
            self._lb.bind("<<ListboxSelect>>", _sel)
        else:
            tk.Label(lf,
                     text="  (chua co nhan nao - hay tao moi)",
                     bg=CARD, fg=GRAY,
                     font=(MONO, 9)).pack(pady=6)
            self._lb = None

        nr = tk.Frame(self, bg=BG)
        nr.pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(nr, text="Tao nhan moi:", bg=BG, fg=GRN2,
                 font=(MONO, 9, "bold")).pack(side=tk.LEFT)

        self._new_ent = tk.Entry(nr, font=(MONO, 11), width=20,
                                 bg=CARD, fg=WHT,
                                 insertbackground=WHT,
                                 relief=tk.FLAT, bd=4)
        self._new_ent.pack(side=tk.LEFT, padx=(8, 0))

        def _typed(evt=None):
            txt = self._new_ent.get().strip()
            if txt:
                self._lbl_var.set(txt)
                if self._lb:
                    self._lb.selection_clear(0, tk.END)
        self._new_ent.bind("<KeyRelease>", _typed)

        tk.Label(self, text="Chia split:",
                 bg=BG, fg=WHT,
                 font=(MONO, 9, "bold")).pack(anchor="w",
                                              padx=20, pady=(12, 2))

        self._split_mode = tk.StringVar(value="auto")
        modes_f = tk.Frame(self, bg=BG)
        modes_f.pack(anchor="w", padx=28)

        rkw = dict(bg=BG, fg=GR2, selectcolor=PANEL,
                   activebackground=BG, activeforeground=WHT,
                   font=(MONO, 9))

        tk.Radiobutton(modes_f,
                       text="Tu dong  (70% train / 15% val / 15% test)",
                       variable=self._split_mode, value="auto",
                       **rkw).pack(anchor="w")
        for sp in SPLITS:
            tk.Radiobutton(modes_f,
                           text=f"Tat ca vao  {sp}",
                           variable=self._split_mode, value=sp,
                           **rkw).pack(anchor="w")

        self._cnt_lbl = tk.Label(self, text="", bg=BG, fg=YEL,
                                 font=(MONO, 8), justify=tk.LEFT)
        self._cnt_lbl.pack(pady=(6, 0), padx=20, anchor="w")

        def _upd(*_):
            lbl = self._lbl_var.get().strip()
            if not lbl:
                self._cnt_lbl.config(text="")
                return
            counts = count_label_clips(self.data_dir, lbl)
            mode   = self._split_mode.get()
            n      = self.n_clips
            if mode == "auto":
                nt = max(1, round(n * 0.7))
                nv = max(0, round(n * 0.15))
                ne = n - nt - nv
                txt = (f"Hien co  train:{counts['train']}  "
                       f"val:{counts['val']}  test:{counts['test']}\n"
                       f"Se them  train:+{nt}  val:+{nv}  test:+{ne}")
            else:
                txt = (f"Hien co  {mode}:{counts[mode]}\n"
                       f"Se them  {mode}:+{n}")
            self._cnt_lbl.config(text=txt)

        self._lbl_var.trace_add("write", _upd)
        self._split_mode.trace_add("write", _upd)

        br = tk.Frame(self, bg=BG)
        br.pack(pady=(14, 16))

        bkw = dict(font=(MONO, 10, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=14, pady=7)
        tk.Button(br, text="Xac nhan",
                  bg=GRN, fg="white",
                  command=self._confirm, **bkw).pack(side=tk.LEFT, padx=8)
        tk.Button(br, text="Huy",
                  bg="#374151", fg=WHT,
                  command=self.destroy, **bkw).pack(side=tk.LEFT, padx=8)

    def _confirm(self):
        label = (self._new_ent.get().strip() or self._lbl_var.get().strip())
        if not label:
            messagebox.showwarning("Thieu nhan",
                                   "Vui long chon hoac nhap ten nhan!",
                                   parent=self)
            return
        label = label.replace(" ", "_")
        self.result = (label, self._split_mode.get())
        self.destroy()


class ProgressWin:
    def __init__(self, root, title="Dang xu ly..."):
        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.configure(bg=BG)
        self.win.geometry("500x130")
        self.win.resizable(False, False)

        tk.Label(self.win, text=title,
                 bg=BG, fg=ACC2,
                 font=(MONO, 12, "bold")).pack(pady=(18, 4))

        self._lbl = tk.Label(self.win, text="Chuan bi...",
                             bg=BG, fg=GR2, font=(MONO, 9))
        self._lbl.pack()

        self._pb = ttk.Progressbar(self.win, mode="indeterminate", length=440)
        self._pb.pack(pady=8)
        self._pb.start(10)

    def update(self, msg):
        try:
            self._lbl.config(text=msg)
            self.win.update_idletasks()
        except Exception:
            pass

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass


class App:
    def __init__(self, root, data_dir, cfg):
        self.root     = root
        self.data_dir = Path(data_dir)
        self.cfg      = cfg

        self.folder_path = None
        self.video_files = []

        self.frames   = []
        self.fps      = 30.0
        self.clips    = []
        self.kept     = []

        self.idx        = 0
        self.play_pos   = 0
        self.is_playing = False
        self._play_job  = None

        self._cfg_vars   = {}
        self._use_margin = tk.BooleanVar(value=True)
        self._use_finger = tk.BooleanVar(value=False)

        root.title(f"VSL Auto Cut  —  {data_dir}")
        root.configure(bg=BG)
        root.geometry("1120x840")
        root.minsize(900, 700)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        tb = tk.Frame(self.root, bg=PANEL, pady=8)
        tb.pack(fill=tk.X)

        tk.Label(tb, text="VSL Auto Cut",
                 bg=PANEL, fg=ACC2,
                 font=(MONO, 14, "bold")).pack(side=tk.LEFT, padx=16)

        bkw = dict(font=(MONO, 9, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=10, pady=4)
        tk.Button(tb, text="Mo Folder",
                  bg=ACC, fg=WHT,
                  command=self._open_folder, **bkw).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Lam moi",
                  bg="#374151", fg=WHT,
                  command=self._refresh_list, **bkw).pack(side=tk.LEFT, padx=4)

        self.lbl_folder = tk.Label(tb, text="Chua chon folder",
                                   bg=PANEL, fg=GRAY, font=(MONO, 8))
        self.lbl_folder.pack(side=tk.LEFT, padx=10)

        # ── Toggle buttons ────────────────────────────────────────────
        tg = tk.Frame(tb, bg=PANEL)
        tg.pack(side=tk.RIGHT, padx=4)

        def _make_toggle(parent, text_on, text_off, var, hint):
            btn_holder = [None]

            def _toggle():
                var.set(not var.get())
                _refresh()

            def _refresh():
                on = var.get()
                btn_holder[0].config(
                    text=f"✔ {text_on}" if on else f"✗ {text_off}",
                    bg=GRN if on else "#374151",
                    fg="white")

            btn = tk.Button(parent, text="", font=(MONO, 8, "bold"),
                            relief=tk.FLAT, cursor="hand2",
                            padx=8, pady=5, command=_toggle)
            btn.pack(pady=2)
            btn_holder[0] = btn
            tk.Label(parent, text=hint, bg=PANEL, fg=GRAY,
                     font=(MONO, 6)).pack()
            _refresh()
            return btn

        tk.Label(tg, text="Bo loc", bg=PANEL, fg=GR2,
                 font=(MONO, 7, "bold")).pack()
        _make_toggle(tg, "Margin tay>vai", "Margin OFF",
                     self._use_margin, "tay phai cao hon vai")
        _make_toggle(tg, "Finger curl", "Finger OFF",
                     self._use_finger, "ngon tay dang co/ky hieu")

        # ── Param sliders + Re-run (tat ca trong pr frame) ───────────
        pr = tk.Frame(tb, bg=PANEL)
        pr.pack(side=tk.RIGHT, padx=12)

        def _make_param(parent, lbl, key, lo, hi, res, default):
            f = tk.Frame(parent, bg=PANEL)
            f.pack(side=tk.LEFT, padx=5)
            tk.Label(f, text=lbl, bg=PANEL, fg=GR2,
                     font=(MONO, 7)).pack()
            var = tk.DoubleVar(value=default)
            tk.Scale(f, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, variable=var,
                     bg=PANEL, fg=WHT, troughcolor=CARD,
                     highlightthickness=0, length=80,
                     font=(MONO, 7), showvalue=True).pack()
            self._cfg_vars[key] = var

        _make_param(pr, "padding(s)", "padding", 0.05, 1.0,  0.05,
                    self.cfg.get("padding", 0.25))
        _make_param(pr, "min_dur(s)", "min_dur", 0.1,  3.0,  0.1,
                    self.cfg.get("min_dur", 0.4))
        _make_param(pr, "idle(s)",    "idle",    0.2,  3.0,  0.1,
                    self.cfg.get("idle_sec", 0.7))
        # FIX: range mo rong xuong -0.5 de bat tay vung bung
        _make_param(pr, "margin",     "margin",  -0.5, 0.15, 0.01,
                    self.cfg.get("margin", 0.0))

        # ── Nut Re-run: dat TRONG pr, SAU khi _make_param xong ───────
        rf = tk.Frame(pr, bg=PANEL)
        rf.pack(side=tk.LEFT, padx=6)
        tk.Label(rf, text="re-analyse", bg=PANEL, fg=GRAY,
                 font=(MONO, 6)).pack()
        tk.Button(rf, text="↻ Re-run",
                  bg=YEL, fg="#1f2937",
                  font=(MONO, 8, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=8, pady=6,
                  command=self._reanalyse).pack()

        # ── Paned layout ──────────────────────────────────────────────
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(pane, bg=BG2, width=270)
        pane.add(left, minsize=200)

        tk.Label(left, text="VIDEO TRONG FOLDER",
                 bg=BG2, fg=GRAY,
                 font=(MONO, 8, "bold")).pack(pady=(8, 2), padx=8, anchor="w")

        vf = tk.Frame(left, bg=CARD)
        vf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 2))
        vsb = tk.Scrollbar(vf)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_lb = tk.Listbox(vf, yscrollcommand=vsb.set,
                                   bg=CARD, fg=WHT,
                                   selectbackground=ACC,
                                   font=(MONO, 9),
                                   relief=tk.FLAT, bd=0,
                                   highlightthickness=0,
                                   activestyle="none")
        self.video_lb.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self.video_lb.yview)
        self.video_lb.bind("<Double-Button-1>", self._on_video_dclick)
        self.video_lb.bind("<Return>",          self._on_video_dclick)

        tk.Button(left, text="Phan tich video nay",
                  bg=YEL, fg="#1f2937",
                  font=(MONO, 9, "bold"), relief=tk.FLAT,
                  cursor="hand2", pady=5,
                  command=self._analyse_selected).pack(
            fill=tk.X, padx=6, pady=2)

        tk.Frame(left, bg=PANEL, height=1).pack(fill=tk.X, padx=6, pady=3)

        tk.Label(left, text="CLIPS PHAT HIEN",
                 bg=BG2, fg=GRAY,
                 font=(MONO, 8, "bold")).pack(pady=(2, 2), padx=8, anchor="w")

        cf = tk.Frame(left, bg=CARD)
        cf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        csb = tk.Scrollbar(cf)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self.clip_lb = tk.Listbox(cf, yscrollcommand=csb.set,
                                  bg=CARD, fg=WHT,
                                  selectbackground=ACC,
                                  font=(MONO, 9),
                                  relief=tk.FLAT, bd=0,
                                  highlightthickness=0,
                                  activestyle="none")
        self.clip_lb.pack(fill=tk.BOTH, expand=True)
        csb.config(command=self.clip_lb.yview)
        self.clip_lb.bind("<<ListboxSelect>>", self._on_clip_select)

        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=600)

        self.canvas = tk.Canvas(right, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#000",
                                highlightthickness=2,
                                highlightbackground=ACC)
        self.canvas.pack(pady=(10, 0), padx=10)
        self._show_placeholder()

        pbr = tk.Frame(right, bg=BG)
        pbr.pack(pady=4)
        pbkw = dict(font=(MONO, 10, "bold"), relief=tk.FLAT,
                    cursor="hand2", padx=8, pady=4)
        self.btn_play = tk.Button(pbr, text="Play",
                                  bg=ACC, fg=WHT,
                                  command=self._toggle_play, **pbkw)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        for lbl, fn in [("<<", self._goto_start),
                        ("<",  lambda: self._step(-1)),
                        (">",  lambda: self._step(1)),
                        (">>", self._goto_end)]:
            tk.Button(pbr, text=lbl, bg="#374151", fg=WHT,
                      command=fn, **pbkw).pack(side=tk.LEFT, padx=2)

        self.scrub_var = tk.IntVar()
        self.scrub = tk.Scale(right, from_=0, to=100,
                              orient=tk.HORIZONTAL, variable=self.scrub_var,
                              bg=BG, fg=WHT, troughcolor=CARD,
                              highlightthickness=0, sliderrelief=tk.FLAT,
                              command=self._on_scrub, length=PREVIEW_W)
        self.scrub.pack(padx=10)
        self.lbl_time = tk.Label(right, text="", bg=BG, fg=GRAY,
                                 font=(MONO, 8))
        self.lbl_time.pack()

        tc = tk.Frame(right, bg=PANEL, padx=10, pady=6)
        tc.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(tc, text="Chinh diem cat:",
                 bg=PANEL, fg=WHT,
                 font=(MONO, 9, "bold")).grid(row=0, column=0,
                                             columnspan=8, sticky="w")

        def _row(row, label, fg_col, entry_var, adj_fn, here_fn):
            tk.Label(tc, text=label, bg=PANEL, fg=fg_col,
                     font=(MONO, 8)).grid(row=row, column=0, padx=4, pady=2)
            tk.Entry(tc, textvariable=entry_var, width=7,
                     bg=CARD, fg=WHT, insertbackground=WHT,
                     font=(MONO, 9), relief=tk.FLAT).grid(
                row=row, column=1, padx=2)
            for ci, (t, d) in enumerate([("-10",-10),("-1",-1),
                                          ("+1",1),("+10",10)]):
                tk.Button(tc, text=t, bg=CARD, fg=WHT,
                          font=(MONO, 7), relief=tk.FLAT, padx=4,
                          command=lambda dd=d: adj_fn(dd)).grid(
                    row=row, column=2+ci, padx=1)
            tk.Button(tc, text="Frame nay", bg=CARD, fg=WHT,
                      font=(MONO, 7), relief=tk.FLAT, padx=6,
                      command=here_fn).grid(row=row, column=6, padx=6)

        self.start_var = tk.IntVar()
        self.end_var   = tk.IntVar()
        _row(1, "Start:", GRN2, self.start_var, self._adj_start, self._set_start_here)
        _row(2, "End:  ", RED2, self.end_var,   self._adj_end,   self._set_end_here)

        ar = tk.Frame(right, bg=BG)
        ar.pack(pady=4)
        akw = dict(font=(MONO, 11, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=12, pady=6)
        self.btn_keep = tk.Button(ar, text="GIU",
                                  bg=GRN, fg="white",
                                  command=self._keep, **akw)
        self.btn_keep.pack(side=tk.LEFT, padx=6)
        self.btn_skip = tk.Button(ar, text="BO",
                                  bg=RED, fg="white",
                                  command=self._skip, **akw)
        self.btn_skip.pack(side=tk.LEFT, padx=6)
        tk.Button(ar, text="< Truoc",
                  bg="#374151", fg=WHT,
                  command=self._prev_clip, **akw).pack(side=tk.LEFT, padx=4)
        self.lbl_nav = tk.Label(ar, text="--",
                                bg=BG, fg=WHT,
                                font=(MONO, 10, "bold"), width=14)
        self.lbl_nav.pack(side=tk.LEFT)
        tk.Button(ar, text="Tiep >",
                  bg="#374151", fg=WHT,
                  command=self._next_clip, **akw).pack(side=tk.LEFT, padx=4)

        sb2 = tk.Frame(self.root, bg=PANEL, pady=8)
        sb2.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = tk.Label(sb2,
                                   text="Chon folder -> chon video -> Phan tich -> Luu",
                                   bg=PANEL, fg=GR2, font=(MONO, 8))
        self.lbl_status.pack(side=tk.LEFT, padx=16)

        tk.Button(sb2, text="LUU  -  Chon nhan & Split",
                  bg=GRN, fg="white",
                  font=(MONO, 11, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=16, pady=6,
                  command=self._save_dialog).pack(side=tk.RIGHT, padx=12)

        self.root.bind("<space>",  lambda e: self._toggle_play())
        self.root.bind("<Left>",   lambda e: self._step(-1))
        self.root.bind("<Right>",  lambda e: self._step(1))
        self.root.bind("<k>",      lambda e: self._keep())
        self.root.bind("<d>",      lambda e: self._skip())
        self.root.bind("<n>",      lambda e: self._next_clip())
        self.root.bind("<p>",      lambda e: self._prev_clip())
        self.root.bind("<Return>", lambda e: self._save_dialog())

    # ------------------------------------------------------------------
    def _open_folder(self):
        path = filedialog.askdirectory(title="Chon folder chua video")
        if not path:
            return
        self.folder_path = Path(path)
        short = str(self.folder_path)
        self.lbl_folder.config(text=short[-65:] if len(short) > 65 else short)
        self._refresh_list()

    def _refresh_list(self):
        if not self.folder_path:
            messagebox.showinfo("Chu y", "Hay chon folder truoc!")
            return
        self.video_files = sorted([
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix in VIDEO_EXTS
        ])
        self.video_lb.delete(0, tk.END)
        for f in self.video_files:
            self.video_lb.insert(tk.END, f"  {f.name}")
        self._set_status(f"Tim thay {len(self.video_files)} video trong folder")

    def _on_video_dclick(self, event=None):
        self._analyse_selected()

    def _analyse_selected(self):
        sel = self.video_lb.curselection()
        if not sel:
            messagebox.showinfo("Chu y", "Hay chon video tu danh sach!")
            return
        self._analyse_video(self.video_files[sel[0]])

    # ------------------------------------------------------------------
    def _reanalyse(self):
        """Chay lai analyze_frames tren frames da load voi params moi tu slider.
        Khong can doc lai video — nhanh hon nhieu."""
        if not self.frames:
            messagebox.showinfo("Chu y",
                                "Chua co frames nao.\nHay phan tich video truoc!")
            return

        margin_val = self._cfg_vars["margin"].get()
        prog = ProgressWin(self.root,
                           f"Re-analyse  margin={margin_val:.2f} ...")
        result = {}

        def worker():
            try:
                clips = analyze_frames(
                    self.frames, self.fps,
                    padding_sec     = self._cfg_vars["padding"].get(),
                    min_dur         = self._cfg_vars["min_dur"].get(),
                    idle_sec        = self._cfg_vars["idle"].get(),
                    shoulder_margin = margin_val,
                    use_margin      = self._use_margin.get(),
                    use_finger      = self._use_finger.get(),
                    progress_cb     = prog.update)
                result["clips"] = clips
            except Exception as ex:
                result["error"] = str(ex)
            finally:
                self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()

        if "error" in result:
            messagebox.showerror("Loi", result["error"])
            return

        self.clips = [[s, e] for s, e in result["clips"]]
        self.kept  = [True] * len(self.clips)

        self.clip_lb.delete(0, tk.END)
        for i, (s, e) in enumerate(self.clips):
            dur = (e - s) / self.fps
            self.clip_lb.insert(tk.END,
                f"  #{i+1:02d}  {s:5d}->{e:5d}  {dur:.2f}s  GIU")

        if self.clips:
            self._load_clip(0)
            self._set_status(
                f"Re-analyse xong: {len(self.clips)} clip  |  "
                f"margin={margin_val:.2f}  "
                f"use_margin={'ON' if self._use_margin.get() else 'OFF'}")
        else:
            self._set_status(
                f"Khong tim thay clip  (margin={margin_val:.2f}) "
                "-- thu giam margin them hoac tat bo loc Margin")
            messagebox.showinfo(
                "Khong tim thay",
                f"Khong phat hien clip nao voi margin={margin_val:.2f}\n\n"
                "Goi y:\n"
                "  • Keo margin xuong -0.3 hoac -0.5\n"
                "  • Tat toggle 'Margin tay>vai'\n"
                "  • Giam min_dur")

    # ------------------------------------------------------------------
    def _analyse_video(self, path):
        self._stop_play()
        self.frames = []
        self.clips  = []
        self.kept   = []
        self.clip_lb.delete(0, tk.END)
        self._show_placeholder()

        prog = ProgressWin(self.root, f"Phan tich: {path.name}")
        result = {}

        def worker():
            try:
                cap    = cv2.VideoCapture(str(path))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    if i % 60 == 0:
                        prog.update(f"Doc frame {i}/{total}...")
                    i += 1
                cap.release()

                clips = analyze_frames(
                    frames, fps,
                    padding_sec     = self._cfg_vars["padding"].get(),
                    min_dur         = self._cfg_vars["min_dur"].get(),
                    idle_sec        = self._cfg_vars["idle"].get(),
                    shoulder_margin = self._cfg_vars["margin"].get(),
                    use_margin      = self._use_margin.get(),
                    use_finger      = self._use_finger.get(),
                    progress_cb     = prog.update)

                result["frames"] = frames
                result["fps"]    = fps
                result["clips"]  = clips
            except Exception as ex:
                result["error"] = str(ex)
            finally:
                self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()

        if "error" in result:
            messagebox.showerror("Loi", result["error"])
            return

        self.frames = result["frames"]
        self.fps    = result["fps"]
        self.clips  = [[s, e] for s, e in result["clips"]]
        self.kept   = [True] * len(self.clips)

        self.clip_lb.delete(0, tk.END)
        for i, (s, e) in enumerate(self.clips):
            dur = (e - s) / self.fps
            self.clip_lb.insert(tk.END,
                f"  #{i+1:02d}  {s:5d}->{e:5d}  {dur:.2f}s  GIU")

        if self.clips:
            self._load_clip(0)
            self._set_status(
                f"Phat hien {len(self.clips)} clip tu {path.name}")
        else:
            self._set_status("Khong tim thay clip -- giam margin hoac min_dur")
            messagebox.showinfo(
                "Khong tim thay",
                "Khong phat hien ky hieu nao.\n\n"
                "Thu:\n"
                "  • Keo slider margin xuong am (vi du -0.2)\n"
                "  • Nhan nut ↻ Re-run\n"
                "  • Hoac tat toggle 'Margin tay>vai'")

    # ------------------------------------------------------------------
    def _on_clip_select(self, event=None):
        sel = self.clip_lb.curselection()
        if sel and sel[0] < len(self.clips):
            self._load_clip(sel[0])

    def _load_clip(self, idx):
        if not self.clips or idx < 0 or idx >= len(self.clips):
            return
        self._stop_play()
        self.idx      = idx
        s, e          = self.clips[idx]
        self.play_pos = s

        self.start_var.set(s)
        self.end_var.set(e)
        self.scrub.configure(from_=s, to=e)
        self.scrub_var.set(s)

        n_kept = sum(self.kept)
        self.lbl_nav.config(text=f"Clip {idx+1}/{len(self.clips)}")
        self._set_status(
            f"Clip #{idx+1}  |  {s}->{e}  ({(e-s)/self.fps:.2f}s)"
            f"  |  {n_kept}/{len(self.clips)} da chon")

        self._update_btn_style()
        self._show_frame(s)
        self._refresh_clip_lb()

    def _update_btn_style(self):
        if not self.clips:
            return
        if self.kept[self.idx]:
            self.btn_keep.config(bg=GRN)
            self.btn_skip.config(bg="#374151")
        else:
            self.btn_keep.config(bg="#374151")
            self.btn_skip.config(bg=RED)

    def _refresh_clip_lb(self):
        for i in range(self.clip_lb.size()):
            if i >= len(self.clips):
                break
            s, e  = self.clips[i]
            dur   = (e - s) / self.fps
            mark  = "GIU" if self.kept[i] else "BO"
            self.clip_lb.delete(i)
            self.clip_lb.insert(i,
                f"  #{i+1:02d}  {s:5d}->{e:5d}  {dur:.2f}s  {mark}")
            bg_c = ACC if i == self.idx else CARD
            self.clip_lb.itemconfig(i, bg=bg_c)
        self.clip_lb.selection_clear(0, tk.END)
        self.clip_lb.selection_set(self.idx)
        self.clip_lb.see(self.idx)

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H,
                                     fill="#080810", outline="")
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="Chon video va nhan Phan tich",
                                fill=GRAY, font=(MONO, 13))

    def _show_frame(self, fidx):
        if not self.frames or not self.clips:
            return
        fidx = max(0, min(fidx, len(self.frames) - 1))
        self.play_pos = fidx
        self.scrub_var.set(fidx)

        s, e = self.clips[self.idx]
        bgr  = self.frames[fidx]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sc   = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * sc), int(h * sc)
        img  = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        y0   = (PREVIEW_H - nh) // 2
        x0   = (PREVIEW_W - nw) // 2
        img[y0:y0+nh, x0:x0+nw] = cv2.resize(rgb, (nw, nh))

        cv2.putText(img,
                    f"Frame {fidx}  +{(fidx-s)/self.fps:.2f}s  "
                    f"/ {(e-s)/self.fps:.2f}s",
                    (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52,
                    (200, 200, 255), 1, cv2.LINE_AA)

        bdr = (16, 185, 129) if self.kept[self.idx] else (239, 68, 68)
        cv2.rectangle(img, (0, 0), (PREVIEW_W-1, PREVIEW_H-1), bdr, 3)

        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas._img = imgtk

        self.lbl_time.config(
            text=f"Frame {fidx}/{e}    "
                 f"({fidx/self.fps:.2f}s / {e/self.fps:.2f}s)    "
                 f"Do dai: {(e-s)/self.fps:.2f}s")

    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self.clips:
            return
        self.is_playing = True
        self.btn_play.config(text="Pause")
        self._play_loop()

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.config(text="Play")
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _play_loop(self):
        if not self.is_playing or not self.clips:
            return
        s, e  = self.clips[self.idx]
        nxt   = self.play_pos + 1
        if nxt > e:
            nxt = s
        self._show_frame(nxt)
        delay = max(1, int(1000 / self.fps))
        self._play_job = self.root.after(delay, self._play_loop)

    def _goto_start(self):
        self._stop_play()
        if self.clips:
            self._show_frame(self.clips[self.idx][0])

    def _goto_end(self):
        self._stop_play()
        if self.clips:
            self._show_frame(self.clips[self.idx][1])

    def _step(self, d):
        self._stop_play()
        if self.clips:
            self._show_frame(self.play_pos + d)

    def _on_scrub(self, val):
        self._stop_play()
        if self.clips:
            self._show_frame(int(float(val)))

    def _apply_trim(self):
        if not self.clips:
            return
        s = int(self.start_var.get())
        e = int(self.end_var.get())
        n = len(self.frames)
        s = max(0, min(s, n - 2))
        e = max(s + 1, min(e, n - 1))
        self.clips[self.idx] = [s, e]
        self.start_var.set(s)
        self.end_var.set(e)
        self.scrub.configure(from_=s, to=e)
        self._show_frame(max(s, min(self.play_pos, e)))
        self._refresh_clip_lb()

    def _adj_start(self, d):
        if not self.clips:
            return
        s, _ = self.clips[self.idx]
        self.start_var.set(max(0, s + d))
        self._apply_trim()

    def _adj_end(self, d):
        if not self.clips:
            return
        _, e = self.clips[self.idx]
        self.end_var.set(min(len(self.frames) - 1, e + d))
        self._apply_trim()

    def _set_start_here(self):
        if self.clips:
            self.start_var.set(self.play_pos)
            self._apply_trim()

    def _set_end_here(self):
        if self.clips:
            self.end_var.set(self.play_pos)
            self._apply_trim()

    def _keep(self):
        if not self.clips:
            return
        self.kept[self.idx] = True
        self._update_btn_style()
        self._refresh_clip_lb()
        if self.idx < len(self.clips) - 1:
            self.root.after(180, lambda: self._load_clip(self.idx + 1))

    def _skip(self):
        if not self.clips:
            return
        self.kept[self.idx] = False
        self._update_btn_style()
        self._refresh_clip_lb()
        if self.idx < len(self.clips) - 1:
            self.root.after(180, lambda: self._load_clip(self.idx + 1))

    def _prev_clip(self):
        self._stop_play()
        if self.clips and self.idx > 0:
            self._load_clip(self.idx - 1)

    def _next_clip(self):
        self._stop_play()
        if self.clips and self.idx < len(self.clips) - 1:
            self._load_clip(self.idx + 1)

    def _save_dialog(self):
        if not self.frames:
            messagebox.showinfo("Chu y", "Chua phan tich video nao!")
            return
        to_save = [(i, s, e) for i, ((s, e), k)
                   in enumerate(zip(self.clips, self.kept)) if k]
        if not to_save:
            messagebox.showwarning("Chu y",
                "Khong co clip nao duoc chon (GIU)!\n"
                "Nhan GIU tren it nhat 1 clip.")
            return

        dlg = LabelDialog(self.root, self.data_dir, len(to_save))
        if dlg.result is None:
            return

        label, split_mode = dlg.result
        self._do_save(to_save, label, split_mode)

    def _do_save(self, to_save, label, split_mode):
        self._stop_play()

        n = len(to_save)
        if split_mode == "auto":
            indices = list(range(n))
            random.shuffle(indices)
            n_train = max(1, round(n * SPLIT_RATIO["train"]))
            n_val   = max(0, round(n * SPLIT_RATIO["val"]))
            n_test  = n - n_train - n_val
            if n_test < 0:
                n_val += n_test
                n_test = 0
            split_assign = {}
            for rank, oi in enumerate(indices):
                if rank < n_train:
                    split_assign[rank] = "train"
                elif rank < n_train + n_val:
                    split_assign[rank] = "val"
                else:
                    split_assign[rank] = "test"
        else:
            split_assign = {i: split_mode for i in range(n)}

        for sp in SPLITS:
            (self.data_dir / sp / label).mkdir(parents=True, exist_ok=True)

        start_idx = next_clip_index(self.data_dir, label)
        h, w      = self.frames[0].shape[:2]
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        saved     = []

        prog = ProgressWin(self.root, "Dang luu clips...")

        def worker():
            for rank, (orig_i, s, e) in enumerate(to_save):
                sp       = split_assign[rank]
                file_idx = start_idx + rank
                name     = f"{label}_{file_idx:04d}.mp4"
                out_path = self.data_dir / sp / label / name

                prog.update(f"Luu {rank+1}/{len(to_save)}: {name}  ->  {sp}/")
                vw = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
                for fi in range(s, e + 1):
                    vw.write(self.frames[fi])
                vw.release()
                saved.append((sp, name))

            self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()

        by_split = {}
        for sp, _ in saved:
            by_split[sp] = by_split.get(sp, 0) + 1

        msg = f"Da luu {len(saved)} clip\n\n"
        msg += f"  Nhan : {label}\n"
        msg += f"  Thu muc: {self.data_dir}\n\n"
        for sp in SPLITS:
            if sp in by_split:
                msg += f"  {sp:5s}: {by_split[sp]} clip\n"

        messagebox.showinfo("Luu thanh cong!", msg)
        self._set_status(
            f"Da luu {len(saved)} clip -> [{label}]  "
            + "  ".join(f"{sp}:{c}" for sp, c in by_split.items()))

        for rank, (orig_i, s, e) in enumerate(to_save):
            self.kept[orig_i] = False
        self._refresh_clip_lb()

    def _set_status(self, msg):
        try:
            self.lbl_status.config(text=msg)
            self.root.update_idletasks()
        except Exception:
            pass


# =====================================================================
# TIME-BASED SPLITTER
# =====================================================================

class TimeSplitApp:

    def __init__(self, root, out_dir):
        self.root    = root
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.folder_path  = None
        self.video_files  = []
        self.cap          = None
        self.video_path   = None
        self.total_frames = 0
        self.fps          = 30.0
        self.duration_sec = 0.0
        self.segments     = []
        self.sel_seg      = 0
        self.play_pos     = 0
        self.is_playing   = False
        self._play_job    = None
        root.title(f"VSL Time Splitter  \u2014  {out_dir}")
        root.configure(bg=BG)
        root.geometry("1120x820")
        root.minsize(900, 680)
        self._build_ui()

    def _build_ui(self):
        tb = tk.Frame(self.root, bg=PANEL, pady=8)
        tb.pack(fill=tk.X)
        tk.Label(tb, text="Time Splitter", bg=PANEL, fg=YEL,
                 font=(MONO, 14, "bold")).pack(side=tk.LEFT, padx=16)
        bkw = dict(font=(MONO, 9, "bold"), relief=tk.FLAT, cursor="hand2", padx=10, pady=4)
        tk.Button(tb, text="Mo Folder", bg=ACC, fg=WHT,
                  command=self._open_folder, **bkw).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Lam moi", bg="#374151", fg=WHT,
                  command=self._refresh_list, **bkw).pack(side=tk.LEFT, padx=4)
        self.lbl_folder = tk.Label(tb, text="Chua chon folder",
                                   bg=PANEL, fg=GRAY, font=(MONO, 8))
        self.lbl_folder.pack(side=tk.LEFT, padx=10)
        cr = tk.Frame(tb, bg=PANEL)
        cr.pack(side=tk.RIGHT, padx=16)
        tk.Label(cr, text="Do dai moi doan (giay):",
                 bg=PANEL, fg=GR2, font=(MONO, 9)).pack(side=tk.LEFT, padx=(0, 6))
        self._chunk_var = tk.DoubleVar(value=3.0)
        tk.Entry(cr, textvariable=self._chunk_var, width=6, font=(MONO, 12),
                 bg=CARD, fg=YEL, insertbackground=YEL,
                 justify=tk.CENTER, relief=tk.FLAT, bd=4).pack(side=tk.LEFT)
        tk.Button(cr, text="Ap dung", bg=YEL, fg="#1f2937",
                  command=self._apply_chunk, **bkw).pack(side=tk.LEFT, padx=6)

        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG, sashwidth=5)
        pane.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(pane, bg=BG2, width=280)
        pane.add(left, minsize=200)
        tk.Label(left, text="VIDEO TRONG FOLDER", bg=BG2, fg=GRAY,
                 font=(MONO, 8, "bold")).pack(pady=(8, 2), padx=8, anchor="w")
        vf = tk.Frame(left, bg=CARD)
        vf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 2))
        vsb = tk.Scrollbar(vf); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_lb = tk.Listbox(vf, yscrollcommand=vsb.set,
                                   bg=CARD, fg=WHT, selectbackground=ACC,
                                   font=(MONO, 9), relief=tk.FLAT, bd=0,
                                   highlightthickness=0, activestyle="none")
        self.video_lb.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self.video_lb.yview)
        self.video_lb.bind("<Double-Button-1>", self._on_video_dclick)
        self.video_lb.bind("<<ListboxSelect>>",  self._on_video_select)
        tk.Frame(left, bg=PANEL, height=1).pack(fill=tk.X, padx=6, pady=3)
        tk.Label(left, text="SEGMENTS", bg=BG2, fg=GRAY,
                 font=(MONO, 8, "bold")).pack(pady=(2, 2), padx=8, anchor="w")
        sf = tk.Frame(left, bg=CARD)
        sf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        ssb = tk.Scrollbar(sf); ssb.pack(side=tk.RIGHT, fill=tk.Y)
        self.seg_lb = tk.Listbox(sf, yscrollcommand=ssb.set,
                                 bg=CARD, fg=WHT, selectbackground=ACC,
                                 font=(MONO, 9), relief=tk.FLAT, bd=0,
                                 highlightthickness=0, activestyle="none")
        self.seg_lb.pack(fill=tk.BOTH, expand=True)
        ssb.config(command=self.seg_lb.yview)
        self.seg_lb.bind("<<ListboxSelect>>", self._on_seg_select)
        togr = tk.Frame(left, bg=BG2)
        togr.pack(fill=tk.X, padx=6, pady=(0, 4))
        tkw = dict(font=(MONO, 8, "bold"), relief=tk.FLAT, cursor="hand2", padx=6, pady=3)
        tk.Button(togr, text="Chon tat ca", bg=GRN, fg="white",
                  command=lambda: self._toggle_all(True), **tkw).pack(side=tk.LEFT, padx=2)
        tk.Button(togr, text="Bo tat ca", bg="#374151", fg=WHT,
                  command=lambda: self._toggle_all(False), **tkw).pack(side=tk.LEFT, padx=2)
        self.lbl_count = tk.Label(togr, text="", bg=BG2, fg=GR2, font=(MONO, 7))
        self.lbl_count.pack(side=tk.LEFT, padx=6)

        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=600)
        self.lbl_info = tk.Label(right, text="Chua chon video", bg=BG, fg=GR2, font=(MONO, 8))
        self.lbl_info.pack(pady=(6, 0))
        self.canvas = tk.Canvas(right, width=PREVIEW_W, height=PREVIEW_H, bg="#000",
                                highlightthickness=2, highlightbackground=YEL)
        self.canvas.pack(pady=(4, 0), padx=10)
        self._show_placeholder("Chon video va nhan Ap dung")
        pbr = tk.Frame(right, bg=BG)
        pbr.pack(pady=4)
        pbkw = dict(font=(MONO, 10, "bold"), relief=tk.FLAT, cursor="hand2", padx=8, pady=4)
        self.btn_play = tk.Button(pbr, text="Play", bg=ACC, fg=WHT,
                                  command=self._toggle_play, **pbkw)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        for lbl, fn in [("<<", self._goto_start), ("<", lambda: self._step(-1)),
                        (">", lambda: self._step(1)), (">>", self._goto_end)]:
            tk.Button(pbr, text=lbl, bg="#374151", fg=WHT,
                      command=fn, **pbkw).pack(side=tk.LEFT, padx=2)
        self.scrub_var = tk.IntVar()
        self.scrub = tk.Scale(right, from_=0, to=100, orient=tk.HORIZONTAL,
                              variable=self.scrub_var, bg=BG, fg=WHT, troughcolor=CARD,
                              highlightthickness=0, sliderrelief=tk.FLAT,
                              command=self._on_scrub, length=PREVIEW_W)
        self.scrub.pack(padx=10)
        self.lbl_time = tk.Label(right, text="", bg=BG, fg=GRAY, font=(MONO, 8))
        self.lbl_time.pack()
        ar = tk.Frame(right, bg=BG)
        ar.pack(pady=4)
        akw = dict(font=(MONO, 11, "bold"), relief=tk.FLAT, cursor="hand2", padx=12, pady=6)
        self.btn_keep = tk.Button(ar, text="GIU", bg=GRN, fg="white",
                                  command=self._keep, **akw)
        self.btn_keep.pack(side=tk.LEFT, padx=6)
        self.btn_skip = tk.Button(ar, text="BO", bg=RED, fg="white",
                                  command=self._skip, **akw)
        self.btn_skip.pack(side=tk.LEFT, padx=6)
        tk.Button(ar, text="< Truoc", bg="#374151", fg=WHT,
                  command=self._prev_seg, **akw).pack(side=tk.LEFT, padx=4)
        self.lbl_nav = tk.Label(ar, text="--", bg=BG, fg=WHT,
                                font=(MONO, 10, "bold"), width=14)
        self.lbl_nav.pack(side=tk.LEFT)
        tk.Button(ar, text="Tiep >", bg="#374151", fg=WHT,
                  command=self._next_seg, **akw).pack(side=tk.LEFT, padx=4)
        sb = tk.Frame(self.root, bg=PANEL, pady=8)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_status = tk.Label(sb, text="Mo folder -> Ap dung -> Luu",
                                   bg=PANEL, fg=GR2, font=(MONO, 8))
        self.lbl_status.pack(side=tk.LEFT, padx=16)
        odr = tk.Frame(sb, bg=PANEL)
        odr.pack(side=tk.LEFT, padx=10)
        tk.Label(odr, text="Luu vao:", bg=PANEL, fg=GRAY, font=(MONO, 8)).pack(side=tk.LEFT)
        self._out_lbl = tk.Label(odr, text=str(self.out_dir)[-55:],
                                 bg=PANEL, fg=YEL, font=(MONO, 8), cursor="hand2")
        self._out_lbl.pack(side=tk.LEFT, padx=4)
        self._out_lbl.bind("<Button-1>", self._choose_out_dir)
        tk.Button(sb, text="LUU  segments da chon",
                  bg=GRN, fg="white",
                  font=(MONO, 11, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=16, pady=6,
                  command=self._save_segments).pack(side=tk.RIGHT, padx=12)
        self.root.bind("<space>",  lambda e: self._toggle_play())
        self.root.bind("<Left>",   lambda e: self._step(-1))
        self.root.bind("<Right>",  lambda e: self._step(1))
        self.root.bind("<k>",      lambda e: self._keep())
        self.root.bind("<d>",      lambda e: self._skip())
        self.root.bind("<n>",      lambda e: self._next_seg())
        self.root.bind("<p>",      lambda e: self._prev_seg())
        self.root.bind("<Return>", lambda e: self._save_segments())

    def _open_folder(self):
        path = filedialog.askdirectory(title="Chon folder chua video")
        if not path:
            return
        self.folder_path = Path(path)
        s = str(self.folder_path)
        self.lbl_folder.config(text=s[-65:] if len(s) > 65 else s)
        self._refresh_list()

    def _refresh_list(self):
        if not self.folder_path:
            messagebox.showinfo("Chu y", "Hay chon folder truoc!")
            return
        self.video_files = sorted([
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix in VIDEO_EXTS
        ])
        self.video_lb.delete(0, tk.END)
        for f in self.video_files:
            self.video_lb.insert(tk.END, f"  {f.name}")
        self._set_status(f"Tim thay {len(self.video_files)} video")

    def _on_video_dclick(self, event=None):
        sel = self.video_lb.curselection()
        if sel:
            self._load_video_meta(self.video_files[sel[0]])
            self._apply_chunk()

    def _on_video_select(self, event=None):
        sel = self.video_lb.curselection()
        if sel:
            self._load_video_meta(self.video_files[sel[0]])

    def _load_video_meta(self, path):
        if self.cap:
            self.cap.release()
        self.cap          = cv2.VideoCapture(str(path))
        self.video_path   = path
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps
        self.segments     = []
        self.sel_seg      = 0
        self.seg_lb.delete(0, tk.END)
        self._stop_play()
        self._show_placeholder(
            f"{path.name}\n{self.duration_sec:.1f}s  |  "
            f"{self.fps:.0f}fps  |  {self.total_frames} frames")
        self.lbl_info.config(
            text=f"{path.name}   {self.duration_sec:.2f}s  "
                 f"{self.fps:.0f}fps  {self.total_frames}f")
        self._set_status(f"Da chon: {path.name} - Nhan Ap dung de cat")

    def _apply_chunk(self):
        if not self.cap or not self.video_path:
            messagebox.showinfo("Chu y", "Hay chon video truoc!")
            return
        try:
            chunk = float(self._chunk_var.get())
            if chunk <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Loi", "Do dai phai la so duong (giay)!")
            return
        chunk_f = int(chunk * self.fps)
        if chunk_f < 1:
            messagebox.showerror("Loi", "Do dai qua nho!")
            return
        self.segments = []
        s = 0
        while s < self.total_frames:
            e   = min(s + chunk_f - 1, self.total_frames - 1)
            dur = (e - s + 1) / self.fps
            if self.segments and dur < chunk * 0.2:
                ps, pe, pk = self.segments[-1]
                self.segments[-1] = (ps, e, pk)
            else:
                self.segments.append((s, e, True))
            s = e + 1
        self._refresh_seg_lb()
        if self.segments:
            self._load_seg(0)
        self._set_status(
            f"Cat thanh {len(self.segments)} doan x {chunk:.1f}s  tu  {self.video_path.name}")

    def _refresh_seg_lb(self):
        self.seg_lb.delete(0, tk.END)
        kept = sum(1 for _, _, k in self.segments if k)
        for i, (s, e, k) in enumerate(self.segments):
            dur  = (e - s + 1) / self.fps
            ts_s = s / self.fps
            mark = "GIU" if k else "BO"
            self.seg_lb.insert(
                tk.END,
                f"  #{i+1:03d}  {ts_s:6.1f}s->{ts_s+dur:6.1f}s  ({dur:.2f}s)  {mark}")
            bg_c = ACC if i == self.sel_seg else (CARD if k else "#1a1a28")
            fg_c = WHT if k else GRAY
            self.seg_lb.itemconfig(i, bg=bg_c, fg=fg_c)
        self.lbl_count.config(text=f"{kept}/{len(self.segments)} chon")

    def _on_seg_select(self, event=None):
        sel = self.seg_lb.curselection()
        if sel and sel[0] < len(self.segments):
            self._load_seg(sel[0])

    def _load_seg(self, idx):
        if not self.segments or idx < 0 or idx >= len(self.segments):
            return
        self._stop_play()
        self.sel_seg  = idx
        s, e, k       = self.segments[idx]
        self.play_pos = s
        self.scrub.configure(from_=s, to=e)
        self.scrub_var.set(s)
        kept = sum(1 for _, _, k2 in self.segments if k2)
        self.lbl_nav.config(text=f"Doan {idx+1}/{len(self.segments)}")
        self._set_status(
            f"Doan #{idx+1}  |  {s/self.fps:.1f}s->{(e+1)/self.fps:.1f}s"
            f"  ({(e-s+1)/self.fps:.2f}s)  |  {kept}/{len(self.segments)} chon")
        self._update_btn_style()
        self._show_frame(s)
        self._refresh_seg_lb()

    def _update_btn_style(self):
        if not self.segments:
            return
        _, _, k = self.segments[self.sel_seg]
        self.btn_keep.config(bg=GRN if k else "#374151")
        self.btn_skip.config(bg=RED if not k else "#374151")

    def _toggle_all(self, state):
        self.segments = [(s, e, state) for s, e, _ in self.segments]
        self._refresh_seg_lb()

    def _read_frame(self, fidx):
        if not self.cap:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _show_placeholder(self, text=""):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H, fill="#080810", outline="")
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text=text or "Chon video",
                                fill=GRAY, font=(MONO, 11), width=PREVIEW_W - 40)

    def _show_frame(self, fidx):
        if not self.segments:
            return
        fidx = max(0, min(fidx, self.total_frames - 1))
        self.play_pos = fidx
        self.scrub_var.set(fidx)
        bgr = self._read_frame(fidx)
        if bgr is None:
            return
        s, e, k = self.segments[self.sel_seg]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sc   = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * sc), int(h * sc)
        img = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        y0 = (PREVIEW_H - nh) // 2
        x0 = (PREVIEW_W - nw) // 2
        img[y0:y0+nh, x0:x0+nw] = cv2.resize(rgb, (nw, nh))
        ts     = fidx / self.fps
        seg_ts = s / self.fps
        cv2.putText(img,
                    f"#{self.sel_seg+1}  {ts:.2f}s  (+{ts-seg_ts:.2f}s / {(e-s+1)/self.fps:.2f}s)",
                    (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 220, 100), 1, cv2.LINE_AA)
        bdr = (16, 185, 129) if k else (239, 68, 68)
        cv2.rectangle(img, (0, 0), (PREVIEW_W - 1, PREVIEW_H - 1), bdr, 3)
        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas._img = imgtk
        self.lbl_time.config(
            text=f"{ts:.2f}s / {self.duration_sec:.2f}s    frame {fidx}/{self.total_frames-1}"
                 f"    doan #{self.sel_seg+1}: {seg_ts:.1f}s->{(e+1)/self.fps:.1f}s")

    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self.segments:
            return
        self.is_playing = True
        self.btn_play.config(text="Pause")
        self._play_loop()

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.config(text="Play")
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _play_loop(self):
        if not self.is_playing or not self.segments:
            return
        s, e, _ = self.segments[self.sel_seg]
        nxt = self.play_pos + 1
        if nxt > e:
            nxt = s
        self._show_frame(nxt)
        self._play_job = self.root.after(max(1, int(1000 / self.fps)), self._play_loop)

    def _goto_start(self):
        self._stop_play()
        if self.segments:
            self._show_frame(self.segments[self.sel_seg][0])

    def _goto_end(self):
        self._stop_play()
        if self.segments:
            self._show_frame(self.segments[self.sel_seg][1])

    def _step(self, d):
        self._stop_play()
        if self.segments:
            self._show_frame(self.play_pos + d)

    def _on_scrub(self, val):
        self._stop_play()
        if self.segments:
            self._show_frame(int(float(val)))

    def _keep(self):
        if not self.segments:
            return
        s, e, _ = self.segments[self.sel_seg]
        self.segments[self.sel_seg] = (s, e, True)
        self._update_btn_style()
        self._refresh_seg_lb()
        if self.sel_seg < len(self.segments) - 1:
            self.root.after(150, lambda: self._load_seg(self.sel_seg + 1))

    def _skip(self):
        if not self.segments:
            return
        s, e, _ = self.segments[self.sel_seg]
        self.segments[self.sel_seg] = (s, e, False)
        self._update_btn_style()
        self._refresh_seg_lb()
        if self.sel_seg < len(self.segments) - 1:
            self.root.after(150, lambda: self._load_seg(self.sel_seg + 1))

    def _prev_seg(self):
        self._stop_play()
        if self.segments and self.sel_seg > 0:
            self._load_seg(self.sel_seg - 1)

    def _next_seg(self):
        self._stop_play()
        if self.segments and self.sel_seg < len(self.segments) - 1:
            self._load_seg(self.sel_seg + 1)

    def _choose_out_dir(self, event=None):
        path = filedialog.askdirectory(title="Chon thu muc luu video")
        if path:
            self.out_dir = Path(path)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            s = str(self.out_dir)
            self._out_lbl.config(text=s[-55:] if len(s) > 55 else s)

    def _save_segments(self):
        to_save = [(i, s, e) for i, (s, e, k) in enumerate(self.segments) if k]
        if not to_save:
            messagebox.showwarning("Chu y", "Khong co doan nao duoc chon!")
            return
        if not self.video_path:
            return
        stem   = self.video_path.stem
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, tmp = self.cap.read()
        if not ret:
            messagebox.showerror("Loi", "Khong doc duoc frame!")
            return
        h, w  = tmp.shape[:2]
        prog  = ProgressWin(self.root, f"Dang luu {len(to_save)} doan...")
        saved = []

        def worker():
            for rank, (orig_i, s, e) in enumerate(to_save):
                name     = f"{stem}_{orig_i+1:04d}.mp4"
                out_path = self.out_dir / name
                prog.update(f"Luu {rank+1}/{len(to_save)}: {name}")
                vw = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, s)
                for _ in range(e - s + 1):
                    ret2, fr = self.cap.read()
                    if not ret2:
                        break
                    vw.write(fr)
                vw.release()
                saved.append(name)
            self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()
        msg = (f"Da luu {len(saved)} doan\n\n"
               f"  Video goc : {self.video_path.name}\n"
               f"  Luu vao   : {self.out_dir}\n\n"
               + "\n".join(f"  {n}" for n in saved[:8])
               + ("\n  ..." if len(saved) > 8 else ""))
        messagebox.showinfo("Luu thanh cong!", msg)
        self._set_status(f"Da luu {len(saved)} doan vao {self.out_dir}")
        for rank, (orig_i, s, e) in enumerate(to_save):
            self.segments[orig_i] = (s, e, False)
        self._refresh_seg_lb()

    def _set_status(self, msg):
        try:
            self.lbl_status.config(text=msg)
            self.root.update_idletasks()
        except Exception:
            pass


# =====================================================================
# LAUNCHER
# =====================================================================

class LauncherApp:
    def __init__(self, root, data_dir, cfg):
        self.root     = root
        self.data_dir = data_dir
        self.cfg      = cfg
        root.title("VSL Video Tools")
        root.configure(bg=BG)
        root.resizable(False, False)
        tk.Label(root, text="VSL Video Tools", bg=BG, fg=ACC2,
                 font=(MONO, 18, "bold")).pack(pady=(32, 4))
        tk.Label(root, text="Chon cong cu ban muon su dung:",
                 bg=BG, fg=GR2, font=(MONO, 10)).pack(pady=(0, 24))
        bkw = dict(font=(MONO, 11, "bold"), relief=tk.FLAT, cursor="hand2", pady=14, width=32)
        f1 = tk.Frame(root, bg=PANEL, padx=20, pady=16)
        f1.pack(padx=40, pady=8, fill=tk.X)
        tk.Button(f1, text="Smart Cut  (MediaPipe)", bg=ACC, fg=WHT,
                  command=self._launch_smart, **bkw).pack()
        tk.Label(f1,
                 text="Phat hien ky hieu tu dong qua pose/tay.\n"
                      "Co cac dieu kien: margin, finger curl, idle...",
                 bg=PANEL, fg=GR2, font=(MONO, 8), justify=tk.LEFT).pack(pady=(6, 0))
        f2 = tk.Frame(root, bg=PANEL, padx=20, pady=16)
        f2.pack(padx=40, pady=8, fill=tk.X)
        tk.Button(f2, text="Time Splitter  (cat theo giay)", bg=YEL, fg="#1f2937",
                  command=self._launch_time, **bkw).pack()
        tk.Label(f2,
                 text="Cat video thanh cac doan deu theo thoi gian ban nhap.\n"
                      "Khong can dieu kien, don gian va nhanh.",
                 bg=PANEL, fg=GR2, font=(MONO, 8), justify=tk.LEFT).pack(pady=(6, 0))
        tk.Label(root, text="", bg=BG).pack(pady=8)

    def _launch_smart(self):
        self.root.destroy()
        r = tk.Tk()
        App(r, self.data_dir, self.cfg)
        r.mainloop()

    def _launch_time(self):
        self.root.destroy()
        r = tk.Tk()
        TimeSplitApp(r, _PROJECT_ROOT / "data" / "raw_splits")
        r.mainloop()


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="VSL Video Tools")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--padding",  type=float, default=0.25)
    ap.add_argument("--min_dur",  type=float, default=0.4)
    ap.add_argument("--idle_sec", type=float, default=0.7)
    ap.add_argument("--margin",   type=float, default=0.0)
    ap.add_argument("--tool",     default="launcher",
                    choices=["launcher", "smart", "time"],
                    help="launcher | smart | time")
    args = ap.parse_args()

    if not HAS_TK:
        print("[ERROR] Can cai: pip install pillow")
        sys.exit(1)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = _PROJECT_ROOT / "data" / "videos"

    data_dir.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        (data_dir / sp).mkdir(exist_ok=True)

    print(f"  Script dir  : {_SCRIPT_DIR}")
    print(f"  Project root: {_PROJECT_ROOT}")
    print(f"  Data dir    : {data_dir}")

    cfg = {"padding": args.padding, "min_dur": args.min_dur,
           "idle_sec": args.idle_sec, "margin": args.margin}

    root = tk.Tk()
    if args.tool == "smart":
        if not HAS_MP:
            print("[ERROR] Can cai: pip install mediapipe")
            sys.exit(1)
        App(root, data_dir, cfg)
    elif args.tool == "time":
        TimeSplitApp(root, _PROJECT_ROOT / "data" / "raw_splits")
    else:
        LauncherApp(root, data_dir, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()