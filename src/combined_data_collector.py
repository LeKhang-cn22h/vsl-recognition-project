"""
VSL Combined Data Collector
Thu thập dữ liệu KẾT HỢP: biểu cảm mặt + hành động tay → 1 ký hiệu hoàn chỉnh
Ví dụ: "ghen_ty" = mặt nheo mắt + tay đặt trước ngực

Dữ liệu lưu mỗi sample: shape (30, 135)
  - 126 features tay (2 tay × 21 điểm × 3 toạ độ)
  -   9 features mặt (EAR, MAR, BROW, NOSE-CHIN, EYE-DIST)
"""

import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data', 'combined_raw')

# ────────────────────────────────────────────────
# Định nghĩa TỪ ĐIỂN KÝ HIỆU KẾT HỢP
# Mỗi từ điển entry gồm:
#   'label'       : tên class (dùng làm tên folder)
#   'description' : mô tả ngắn để người thu thập biết phải làm gì
# ────────────────────────────────────────────────
COMBINED_SIGNS = [
    {
        'label':       'ghen_ty',
        'description': 'Mặt nheo mắt/cau mày, tay đặt chéo trước ngực',
    },
    {
        'label':       'buon_ba',
        'description': 'Mặt buồn, khoé miệng xuống, tay đặt lên ngực/tim',
    },
    {
        'label':       'vui_mung',
        'description': 'Mặt cười, tay vỗ tay hoặc giơ lên cao',
    },
    {
        'label':       'tuc_gian',
        'description': 'Mặt cau mày, mắt nhỏ, tay nắm đấm hoặc chỉ thẳng',
    },
    {
        'label':       'so_hai',
        'description': 'Mặt ngạc nhiên/mắt mở to, tay che miệng hoặc giơ lên',
    },
    {
        'label':       'yeu_thuong',
        'description': 'Mặt tươi/mỉm cười, 2 tay bắt chéo trước ngực (ôm)',
    },
    {
        'label':       'met_moi',
        'description': 'Mặt mệt mỏi, mắt nhắm/hơi nhắm, tay đặt lên trán hoặc má',
    },
    {
        'label':       'xin_loi',
        'description': 'Mặt hối lỗi/buồn, tay nắm đấm xoa vòng tròn trước ngực',
    },
    {
        'label':       'cam_on',
        'description': 'Mặt cười, tay chạm cằm rồi đưa ra phía trước',
    },
    {
        'label':       'khong_biet',
        'description': 'Mặt ngơ ngác/nhíu mày, tay giơ lên lắc nhẹ (shrug)',
    },
]

SIGN_LABELS = [s['label'] for s in COMBINED_SIGNS]

SEQUENCE_LENGTH = 30   # frames mỗi sample
N_HAND_FEATURES = 126  # 2 tay × 21 × 3
N_FACE_FEATURES = 9    # EAR×2, avgEAR, MAR, BROW×2, avgBROW, nose_chin, eye_dist
N_TOTAL         = N_HAND_FEATURES + N_FACE_FEATURES   # 135


# ══════════════════════════════════════════════════════
class CombinedDataCollector:

    # MediaPipe face landmark indices
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX     = [61, 291, 0, 17, 269, 405]
    L_BROW_IDX    = [70, 63, 105, 66, 107]
    R_BROW_IDX    = [300, 293, 334, 296, 336]

    HAND_CONNS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    def __init__(self, output_dir=DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        for sign in SIGN_LABELS:
            os.makedirs(os.path.join(output_dir, sign), exist_ok=True)

        self._init_models()
        print("✓ CombinedDataCollector khởi tạo xong")

    # ──────────────────────────────────────────── init
    def _find_or_download(self, filename, url):
        paths = [
            os.path.join(BASE_DIR, filename),
            os.path.join(BASE_DIR, '..', filename),
            os.path.join(os.getcwd(), filename),
        ]
        for p in paths:
            if os.path.exists(p):
                return os.path.abspath(p)
        save = os.path.join(BASE_DIR, filename)
        print(f"  Downloading {filename}...")
        import urllib.request
        def _prog(c, bs, ts):
            if ts > 0:
                print(f"\r  {min(int(c*bs*100/ts),100)}%", end='', flush=True)
        urllib.request.urlretrieve(url, save, reporthook=_prog)
        print(f"\n  ✓ Saved: {save}")
        return save

    def _init_models(self):
        print("[Init] Hand Landmarker...")
        hand_path = self._find_or_download(
            'hand_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_path),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        print("[Init] Face Landmarker...")
        face_path = self._find_or_download(
            'face_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        print("[Init] ✓ Tất cả models đã sẵn sàng")

    # ──────────────────────────────────────────── feature extraction

    def _dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _ear(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2.0*h) if h > 0 else 0.0

    def _mar(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2.0*h) if h > 0 else 0.0

    def _brow_pos(self, brow_pts, eye_pts):
        return np.mean([p[1] for p in eye_pts]) - np.mean([p[1] for p in brow_pts])

    def extract_face_features(self, face_landmarks):
        """Trích 9 facial features → np.array(9,)"""
        if not face_landmarks:
            return np.zeros(N_FACE_FEATURES, dtype=np.float32)
        lm = face_landmarks[0]
        xy = lambda i: (lm[i].x, lm[i].y)
        le = [xy(i) for i in self.LEFT_EYE_IDX]
        re = [xy(i) for i in self.RIGHT_EYE_IDX]
        mo = [xy(i) for i in self.MOUTH_IDX]
        lb = [xy(i) for i in self.L_BROW_IDX]
        rb = [xy(i) for i in self.R_BROW_IDX]
        left_ear  = self._ear(le);  right_ear = self._ear(re)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = self._mar(mo)
        left_brow  = self._brow_pos(lb, le); right_brow = self._brow_pos(rb, re)
        avg_brow   = (left_brow + right_brow) / 2.0
        nose = xy(1);  chin = xy(152)
        leye = xy(33); reye = xy(263)
        eye_dist  = self._dist(leye, reye)
        nose_chin = self._dist(nose, chin) / (eye_dist + 1e-6)
        return np.array([left_ear, right_ear, avg_ear, mar,
                         left_brow, right_brow, avg_brow,
                         nose_chin, eye_dist], dtype=np.float32)

    def extract_hand_features(self, hand_result):
        """Trích 126 hand features, chuẩn hoá theo cổ tay → list(126)"""
        kps = []
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                for lm in hand_lm:
                    kps.extend([lm.x, lm.y, lm.z])
        else:
            kps = [0.0] * 63
        while len(kps) < N_HAND_FEATURES:
            kps.extend([0.0] * 63)
        kps = kps[:N_HAND_FEATURES]

        # Normalise per hand (relative to wrist)
        arr = np.array(kps).reshape(-1, 3)
        for i in range(2):
            s, e = i*21, i*21+21
            h_kps = arr[s:e]
            if np.any(h_kps != 0):
                h_kps -= h_kps[0].copy()
                arr[s:e] = h_kps
        return arr.flatten().tolist()

    def extract_combined_frame(self, hand_result, face_landmarks):
        """Tổng hợp 1 frame = 135 features"""
        hand_feat = self.extract_hand_features(hand_result)          # 126
        face_feat = self.extract_face_features(face_landmarks).tolist()  # 9
        return hand_feat + face_feat   # 135

    # ──────────────────────────────────────────── draw helpers

    def _draw_hands(self, frame, hand_result):
        if not hand_result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lm in hand_result.hand_landmarks:
            for lm in hand_lm:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,0), -1)
            for a, b in self.HAND_CONNS:
                pa = (int(hand_lm[a].x*w), int(hand_lm[a].y*h))
                pb = (int(hand_lm[b].x*w), int(hand_lm[b].y*h))
                cv2.line(frame, pa, pb, (200,200,200), 2)
        return frame

    def _draw_face(self, frame, face_result):
        if not face_result.face_landmarks:
            return frame
        h, w = frame.shape[:2]
        for face_lm in face_result.face_landmarks:
            xs = [lm.x*w for lm in face_lm]; ys = [lm.y*h for lm in face_lm]
            cv2.rectangle(frame, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), (0,180,255), 2)
            for idx in [33,133,362,263,61,291,199]:
                if idx < len(face_lm):
                    cv2.circle(frame,
                               (int(face_lm[idx].x*w), int(face_lm[idx].y*h)),
                               3, (0,255,255), -1)
        return frame

    # ──────────────────────────────────────────── collect

    def collect_sign(self, sign_info: dict, num_samples: int = 50):
        label       = sign_info['label']
        description = sign_info['description']
        sign_dir    = os.path.join(self.output_dir, label)
        existing    = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])

        # Mở camera
        cap = None
        for idx in [1, 2]:
            test = cv2.VideoCapture(idx)
            if test.isOpened():
                ret, _ = test.read()
                if ret:
                    cap = test
                    print(f"  Dùng camera index {idx}")
                    break
                test.release()
            else:
                test.release()
        if cap is None:
            print("  ✗ Không mở được camera!")
            return 0

        sample_count = existing
        recording    = False
        auto_mode    = False   # ấn A: tự động ghi liên tục
        sequence     = []
        rest_frames  = 0       # đếm frame nghỉ giữa 2 mẫu auto

        AUTO_REST_FRAMES = 10  # số frame chờ giữa mỗi mẫu auto (~0.3s ở 30fps)

        print(f"\n{'='*65}")
        print(f"  Ký hiệu : {label.upper()}")
        print(f"  Hướng dẫn: {description}")
        print(f"  Hiện có  : {existing} mẫu | Cần thêm: {num_samples}")
        print(f"{'='*65}")
        print("  S: ghi 1 mẫu  |  A: tự động ghi hết  |  Q: thoát\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result  = self.hand_detector.detect(mp_img)
            face_result  = self.face_mesh.detect(mp_img)

            frame = self._draw_hands(frame, hand_result)
            frame = self._draw_face(frame, face_result)

            hand_ok = bool(hand_result.hand_landmarks)
            face_ok = bool(face_result.face_landmarks)

            h_frame, w_frame = frame.shape[:2]

            # ── Auto mode: kích hoạt recording sau khi nghỉ đủ frame ──
            if auto_mode and not recording:
                if rest_frames > 0:
                    rest_frames -= 1
                else:
                    recording = True
                    sequence  = []

            # ── Ghi sequence khi đang recording ──
            if recording:
                combined = self.extract_combined_frame(
                    hand_result, face_result.face_landmarks
                )
                sequence.append(combined)

                if len(sequence) >= SEQUENCE_LENGTH:
                    path = os.path.join(sign_dir, f'sample_{sample_count:04d}.npy')
                    np.save(path, np.array(sequence, dtype=np.float32))
                    sample_count += 1
                    sequence  = []
                    recording = False
                    print(f"  ✓ Mẫu {sample_count}/{existing+num_samples}  [{label}]"
                          + ("  [AUTO]" if auto_mode else ""))

                    if (sample_count - existing) >= num_samples:
                        print(f"  ✓ Đã đủ {num_samples} mẫu mới!")
                        auto_mode = False
                        break

                    # Nếu auto: đặt lại thời gian nghỉ trước mẫu tiếp theo
                    if auto_mode:
                        rest_frames = AUTO_REST_FRAMES

            # ── HUD ──
            cv2.rectangle(frame, (0, 0), (w_frame, 165), (0,0,0), -1)

            # Tiêu đề + màu khác nhau giữa manual / auto
            title_color = (0, 255, 100) if auto_mode else (0, 220, 255)
            mode_label  = "  [AUTO]" if auto_mode else ""
            cv2.putText(frame, label.upper() + mode_label, (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, title_color, 3)

            desc_lines = [description[i:i+55] for i in range(0, len(description), 55)]
            for li, dl in enumerate(desc_lines[:2]):
                cv2.putText(frame, dl, (12, 80 + li*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1)

            cv2.putText(frame,
                        f"Mau: {sample_count-existing}/{num_samples}  (Tong:{sample_count})",
                        (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            # Trạng thái tay / mặt
            h_col = (0,255,0) if hand_ok else (0,0,255)
            f_col = (0,255,0) if face_ok else (0,0,255)
            cv2.putText(frame, f"Tay:{'OK' if hand_ok else 'X'}",
                        (12, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, h_col, 2)
            cv2.putText(frame, f"Mat:{'OK' if face_ok else 'X'}",
                        (120, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, f_col, 2)

            # Thanh tiến trình recording / rest
            if recording:
                prog = len(sequence) / SEQUENCE_LENGTH
                bx, by, bw = 12, 175, w_frame - 24
                bar_color = (0, 200, 50) if auto_mode else (0, 220, 0)
                cv2.rectangle(frame, (bx, by), (bx+bw, by+16), (50,50,50), -1)
                cv2.rectangle(frame, (bx, by), (bx+int(bw*prog), by+16), bar_color, -1)
                cv2.putText(frame, f"RECORDING... {len(sequence)}/{SEQUENCE_LENGTH}",
                            (bx, by-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)
            elif auto_mode and rest_frames > 0:
                # Hiển thị countdown nghỉ giữa các mẫu auto
                prog = 1.0 - rest_frames / AUTO_REST_FRAMES
                bx, by, bw = 12, 175, w_frame - 24
                cv2.rectangle(frame, (bx, by), (bx+bw, by+16), (50,50,50), -1)
                cv2.rectangle(frame, (bx, by), (bx+int(bw*prog), by+16), (0,140,255), -1)
                cv2.putText(frame, "Chuan bi...",
                            (bx, by-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 1)

            hint = "S:Ghi  A:Tu dong  Q:Thoat" if not auto_mode else "A:Dung lai  Q:Thoat"
            cv2.putText(frame, hint,
                        (w_frame - 270, h_frame - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,130), 1)

            cv2.imshow(f'Combined Collector - {label}', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                if not auto_mode and not recording:
                    recording = True
                    sequence  = []
                    print(f"  → Ghi mẫu {sample_count+1}...")
            elif key in (ord('a'), ord('A')):
                if not auto_mode:
                    auto_mode   = True
                    rest_frames = 0
                    print(f"  → AUTO MODE bật: sẽ tự động ghi {num_samples-(sample_count-existing)} mẫu còn lại...")
                else:
                    auto_mode = False
                    recording = False
                    sequence  = []
                    print("  → AUTO MODE tắt.")
            elif key in (ord('q'), ord('Q')):
                break

        cap.release()
        cv2.destroyAllWindows()
        collected = sample_count - existing
        print(f"\n  Thu thập: {collected} mẫu mới cho '{label}'. Tổng: {sample_count}")
        return collected


def main():
    print("\n" + "="*65)
    print("  VSL COMBINED DATA COLLECTOR (Mặt + Tay)")
    print("="*65)

    collector = CombinedDataCollector()

    print("\n  Danh sách ký hiệu có thể thu thập:")
    for i, s in enumerate(COMBINED_SIGNS, 1):
        existing = len([f for f in os.listdir(os.path.join(DATA_DIR, s['label']))
                        if f.endswith('.npy')]) if os.path.exists(
                            os.path.join(DATA_DIR, s['label'])) else 0
        print(f"  [{i:2d}] {s['label']:<15} → {s['description']:<50}  ({existing} mẫu)")

    print("\n  Nhập số thứ tự để thu thập, hoặc 'all' cho tất cả, Enter để thoát.")

    while True:
        choice = input("\n  Chọn (số/all/Enter thoát): ").strip().lower()
        if choice == "":
            break
        elif choice == "all":
            targets = COMBINED_SIGNS
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(COMBINED_SIGNS):
                    targets = [COMBINED_SIGNS[idx]]
                else:
                    print("  Số không hợp lệ!")
                    continue
            except ValueError:
                print("  Nhập số hoặc 'all'!")
                continue

        try:
            n_str = input("  Số mẫu cần thu thập (mặc định 50): ").strip()
            n = int(n_str) if n_str else 50
        except ValueError:
            n = 50

        for sign_info in targets:
            collector.collect_sign(sign_info, num_samples=n)

    print(f"\n  ✓ Hoàn thành! Data tại: {os.path.abspath(DATA_DIR)}")
    input("  Enter để thoát...")


if __name__ == '__main__':
    main()