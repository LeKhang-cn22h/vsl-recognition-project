"""
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
"""

import cv2
import json
import os
import time
import urllib.request
import zipfile
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from collector import (
    init_hf, upload_to_hf,
    FullBodyDrawer, draw_text_bg, lm_to_px,
    FramingChecker,
    FacialExpressionAnalyzer,
    InteractionVisualizer,
)

# Import flush_hf — tương thích với hf_uploader.py (queue-based)
_flush_hf = None
for _mod in ('collector.hf_upload', 'collector.hf_uploader', 'hf_uploader'):
    try:
        import importlib as _il
        _m = _il.import_module(_mod)
        if hasattr(_m, 'flush_hf'):
            _flush_hf = _m.flush_hf
            break
    except ImportError:
        pass

def _do_flush(commit_msg: str = None):
    """Gọi flush để thực sự commit queue lên HuggingFace (nếu dùng queue-based uploader)."""
    if _flush_hf is not None:
        try:
            n = _flush_hf(commit_msg)
            if n and n > 0:
                print(f"  [HF] Commit thanh cong: {n} file!")
        except Exception as e:
            print(f"  [HF] Flush error: {e}")

# ── Download MediaPipe models ──────────────────────────────

MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
}

def download_model(filename):
    if os.path.exists(filename):
        return filename
    print(f"  Dang tai {filename} ...")
    urllib.request.urlretrieve(MODEL_URLS[filename], filename)
    return filename


# ── Emotion constants (đồng bộ với video_to_npy.py) ──────

EMOTIONS = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "sad":      4,
    "surprise": 5,
    "neutral":  6,
}
EMOTIONS_LIST = list(EMOTIONS.keys())


# ── Emotion helpers ───────────────────────────────────────

def ask_emotion(default: str = None) -> str:
    """
    Hiển thị menu chọn emotion.
    Trả về tên emotion (str) hoặc None nếu bỏ qua.
    """
    print("\n  ┌─ CHON EMOTION CHO NHAN NAY ─────────────────────────────┐")
    for i, name in enumerate(EMOTIONS_LIST, 1):
        marker = " ◀ mac dinh" if name == default else ""
        print(f"  │  {i}. {name}{marker}")
    print("  │  0. Bo qua (khong gan emotion)")
    print("  └──────────────────────────────────────────────────────────┘")

    prompt = f"  Chon (1-{len(EMOTIONS_LIST)}"
    if default:
        prompt += f", Enter='{default}'"
    prompt += ", 0=bo qua): "

    while True:
        raw = input(prompt).strip().lower()
        if raw == "" and default:
            print(f"  → Dung mac dinh: {default}")
            return default
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(EMOTIONS_LIST):
            chosen = EMOTIONS_LIST[int(raw) - 1]
            print(f"  → Emotion: {chosen}")
            return chosen
        if raw in EMOTIONS:
            print(f"  → Emotion: {raw}")
            return raw
        print(f"  Khong hop le! Nhap so tu 1-{len(EMOTIONS_LIST)} hoac ten emotion.")


def save_video_emotion(video_path: str, emotion: str):
    """Lưu emotion vào file .json cạnh video (tương thích video_to_npy.py)."""
    if not emotion:
        return
    meta_path = os.path.splitext(video_path)[0] + ".json"
    data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            pass
    data['emotion']         = emotion
    data['emotion_id']      = EMOTIONS.get(emotion, 0)
    data['emotion_updated'] = datetime.now().isoformat()
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_video_emotion(video_path: str):
    """Đọc emotion từ file .json cạnh video."""
    meta_path = os.path.splitext(video_path)[0] + ".json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('emotion')
        except Exception:
            pass
    return None


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
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_pose))

        print("  Khoi tao HandLandmarker ...")
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
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

        cap = cv2.VideoCapture(1)
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

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()


def main():
    collector = WebcamVideoCollector(output_dir='data/videos')
    try:
        collector.interactive_menu()
    finally:
        collector.close()


if __name__ == "__main__":
    main()