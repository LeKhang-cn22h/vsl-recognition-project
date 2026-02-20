"""
VSL Combined Real-time Tester
Nhận diện ký hiệu kết hợp: biểu cảm mặt + hành động tay → ý nghĩa
Dùng model combined_model.h5 đã train từ combined_trainer.py
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque


@keras.utils.register_keras_serializable(package='VSL')
class FeatureSlice(layers.Layer):
    """Phải khai báo lại ở đây để load model không bị lỗi unknown layer."""
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
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, '..', 'models')

SEQUENCE_LENGTH  = 30
N_HAND_FEATURES  = 126
N_FACE_FEATURES  = 9
N_TOTAL_FEATURES = N_HAND_FEATURES + N_FACE_FEATURES   # 135


# ══════════════════════════════════════════════════════ Meaning Map
# Thêm/sửa theo từ vựng của bạn

SIGN_MEANINGS = {
    'ghen_ty':    "Tôi đang ghen tị",
    'buon_ba':    "Tôi buồn / đau lòng",
    'vui_mung':   "Tôi vui mừng!",
    'tuc_gian':   "Tôi tức giận",
    'so_hai':     "Tôi sợ hãi!",
    'yeu_thuong': "Tôi yêu bạn ❤",
    'met_moi':    "Tôi mệt mỏi",
    'xin_loi':    "Xin lỗi bạn",
    'cam_on':     "Cảm ơn bạn",
    'khong_biet': "Tôi không biết / không chắc",
}

SIGN_COLORS = {
    'ghen_ty':    (128,   0, 128),
    'buon_ba':    (255, 100, 100),
    'vui_mung':   (  0, 220,  50),
    'tuc_gian':   (  0,   0, 220),
    'so_hai':     (  0, 200, 220),
    'yeu_thuong': (  0, 100, 255),
    'met_moi':    (150, 150, 150),
    'xin_loi':    (  0, 165, 255),
    'cam_on':     ( 50, 200, 100),
    'khong_biet': (200, 200, 200),
}

HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


# ══════════════════════════════════════════════════════ Tester

class CombinedTester:

    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX     = [61, 291, 0, 17, 269, 405]
    L_BROW_IDX    = [70, 63, 105, 66, 107]
    R_BROW_IDX    = [300, 293, 334, 296, 336]

    def __init__(self):
        self._load_model()
        self._init_mediapipe()
        self.buffer          = deque(maxlen=SEQUENCE_LENGTH)
        self._sign_buf       = deque(maxlen=7)
        self._meaning_hist   = deque(maxlen=5)
        self._current_sign   = None
        self._current_conf   = 0.0
        self._current_meaning = ""
        self._last_committed = None
        print("\n✓ CombinedTester sẵn sàng!\n")

    # ──────────────────────────────────────────── load
    def _load_model(self):
        # Ưu tiên .keras (mới), fallback .h5 (cũ)
        keras_path = os.path.join(MODEL_DIR, 'combined_model.keras')
        h5_path    = os.path.join(MODEL_DIR, 'combined_model.h5')
        model_path = keras_path if os.path.exists(keras_path) else h5_path
        encoder_path = os.path.join(MODEL_DIR, 'combined_label_encoder.npy')
        meta_path    = os.path.join(MODEL_DIR, 'combined_model_meta.json')

        if not os.path.exists(model_path):
            print(f"✗ Không tìm thấy: {model_path}")
            print("  → Hãy chạy combined_trainer.py trước!")
            raise FileNotFoundError(model_path)
        if not os.path.exists(encoder_path):
            print(f"✗ Không tìm thấy: {encoder_path}")
            raise FileNotFoundError(encoder_path)

        print(f"[Load] Model: {model_path}")
        self.model  = tf.keras.models.load_model(model_path)
        self.labels = np.load(encoder_path, allow_pickle=True)

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
            print(f"[Load] Labels ({len(self.labels)}): {self.labels}")
            print(f"[Load] Model type: {self.meta.get('model_type','?')}")
        else:
            self.meta = {}

    # ──────────────────────────────────────────── mediapipe
    def _download(self, filename, url):
        import urllib.request
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"  ✓ Done")

    def _init_mediapipe(self):
        hand_task = 'hand_landmarker.task'
        face_task = 'face_landmarker.task'

        for fn, url in [
            (hand_task, 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
            (face_task, 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
        ]:
            if not os.path.exists(fn):
                # Cũng tìm trong BASE_DIR
                alt = os.path.join(BASE_DIR, fn)
                if os.path.exists(alt):
                    fn = alt
                else:
                    self._download(fn, url)

        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_task),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_task),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    # ──────────────────────────────────────────── features
    def _dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _ear(self, pts):
        v1 = self._dist(pts[1], pts[5]); v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2*h) if h > 0 else 0

    def _mar(self, pts):
        v1 = self._dist(pts[1], pts[5]); v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2*h) if h > 0 else 0

    def _brow(self, b, e):
        return np.mean([p[1] for p in e]) - np.mean([p[1] for p in b])

    def _face_feat(self, face_lms):
        if not face_lms:
            return [0.0] * N_FACE_FEATURES
        lm = face_lms[0]
        xy = lambda i: (lm[i].x, lm[i].y)
        le = [xy(i) for i in self.LEFT_EYE_IDX]
        re = [xy(i) for i in self.RIGHT_EYE_IDX]
        mo = [xy(i) for i in self.MOUTH_IDX]
        lb = [xy(i) for i in self.L_BROW_IDX]
        rb = [xy(i) for i in self.R_BROW_IDX]
        l_ear = self._ear(le); r_ear = self._ear(re); avg_ear = (l_ear+r_ear)/2
        mar   = self._mar(mo)
        lb_pos = self._brow(lb, le); rb_pos = self._brow(rb, re)
        avg_brow = (lb_pos+rb_pos)/2
        nose = xy(1); chin = xy(152); leye = xy(33); reye = xy(263)
        ed = self._dist(leye, reye)
        nc = self._dist(nose, chin) / (ed + 1e-6)
        return [l_ear, r_ear, avg_ear, mar, lb_pos, rb_pos, avg_brow, nc, ed]

    def _hand_feat(self, hand_result):
        kps = []
        if hand_result.hand_landmarks:
            for h_lm in hand_result.hand_landmarks:
                for lm in h_lm:
                    kps.extend([lm.x, lm.y, lm.z])
        else:
            kps = [0.0]*63
        while len(kps) < N_HAND_FEATURES:
            kps.extend([0.0]*63)
        kps = kps[:N_HAND_FEATURES]
        arr = np.array(kps).reshape(-1, 3)
        for i in range(2):
            s, e = i*21, i*21+21
            h = arr[s:e]
            if np.any(h != 0):
                h -= h[0].copy(); arr[s:e] = h
        return arr.flatten().tolist()

    def _combined_frame(self, hand_result, face_lms):
        return self._hand_feat(hand_result) + self._face_feat(face_lms)

    # ──────────────────────────────────────────── predict
    def _predict(self):
        if len(self.buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        seq  = np.expand_dims(np.array(list(self.buffer), dtype=np.float32), 0)
        pred = self.model.predict(seq, verbose=0)[0]
        conf = float(np.max(pred))
        idx  = int(np.argmax(pred))
        if conf < 0.65:
            return None, conf
        raw = self.labels[idx]
        self._sign_buf.append(raw)
        sign = max(set(self._sign_buf), key=list(self._sign_buf).count)
        return sign, conf

    # ──────────────────────────────────────────── draw
    def _draw_hands(self, frame, hand_result):
        if not hand_result.hand_landmarks: return frame
        h, w = frame.shape[:2]
        for hlm in hand_result.hand_landmarks:
            for lm in hlm:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,60), -1)
            for a, b in HAND_CONNS:
                cv2.line(frame,
                         (int(hlm[a].x*w), int(hlm[a].y*h)),
                         (int(hlm[b].x*w), int(hlm[b].y*h)),
                         (200,200,200), 2)
        return frame

    def _draw_face(self, frame, face_result, sign):
        if not face_result.face_landmarks: return frame
        h, w = frame.shape[:2]
        color = SIGN_COLORS.get(sign or '', (0,180,255))
        for flm in face_result.face_landmarks:
            xs = [l.x*w for l in flm]; ys = [l.y*h for l in flm]
            x1,x2 = int(min(xs)), int(max(xs)); y1,y2 = int(min(ys)), int(max(ys))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            for idx in [33,133,362,263,61,291,199]:
                if idx < len(flm):
                    cv2.circle(frame, (int(flm[idx].x*w), int(flm[idx].y*h)),
                               3, (0,255,255), -1)
        return frame

    def _draw_hud(self, frame, sign, conf, meaning):
        H, W = frame.shape[:2]
        color = SIGN_COLORS.get(sign or '', (200,200,200))

        # ─ Top bar ─
        cv2.rectangle(frame, (0,0), (W, 145), (0,0,0), -1)

        if sign:
            cv2.putText(frame, sign.upper().replace('_', ' '),
                        (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
            conf_bar_w = int((W-28) * conf)
            cv2.rectangle(frame, (14, 70), (14+conf_bar_w, 82), color, -1)
            cv2.putText(frame, f"Confidence: {conf:.0%}",
                        (14, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        else:
            cv2.putText(frame, "Dang nhan dang...",
                        (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (140,140,140), 2)

        # ─ Buffer bar ─
        buf = len(self.buffer)
        bx, by, bw = 14, 120, W-28
        cv2.rectangle(frame, (bx, by), (bx+bw, by+14), (40,40,40), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(bw*buf/SEQUENCE_LENGTH), by+14),
                      (0,180,100), -1)
        cv2.putText(frame, f"Buffer {buf}/{SEQUENCE_LENGTH}",
                    (bx+5, by+11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        # ─ Meaning box (giữa màn hình) ─
        if meaning:
            (tw, th), _ = cv2.getTextSize(meaning, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
            mx = W//2 - tw//2; my = H//2
            cv2.rectangle(frame, (mx-15, my-th-12), (mx+tw+15, my+10), (0,0,0), -1)
            cv2.rectangle(frame, (mx-15, my-th-12), (mx+tw+15, my+10), color, 2)
            cv2.putText(frame, meaning, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)

        # ─ History ─
        cv2.rectangle(frame, (0, H-120), (W, H), (0,0,0), -1)
        cv2.putText(frame, "Lich su:", (10, H-102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,80,80), 1)
        for i, past_sign, past_meaning in [
            (i, *v) for i, v in enumerate(reversed(list(self._meaning_hist)))
        ][:4]:
            alpha = max(255 - i*55, 80)
            pc = SIGN_COLORS.get(past_sign, (180,180,180))
            cv2.putText(frame,
                        f"{past_sign.upper().replace('_',' ')}: {past_meaning}",
                        (10, H-82 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, pc, 1)

        # ─ Controls hint ─
        cv2.putText(frame, "Q: thoat  C: xoa lich su  SPACE: reset buffer",
                    (10, H-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
        return frame

    # ──────────────────────────────────────────── run
    def run(self):
        cap = None
        for idx in [1, 2]:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    cap = c
                    print(f"  Camera: index {idx}")
                    break
                c.release()
            else:
                c.release()
        if cap is None:
            print("✗ Không mở được camera!")
            return

        print("="*55)
        print("  VSL COMBINED REAL-TIME TESTER")
        print("  Làm ký hiệu (mặt + tay) trong ~2 giây để nhận diện")
        print("="*55 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = self.hand_detector.detect(mp_img)
            face_result = self.face_mesh.detect(mp_img)

            # Tích lũy frame vào buffer
            frame_feat = self._combined_frame(hand_result, face_result.face_landmarks)
            self.buffer.append(frame_feat)

            # Dự đoán
            sign, conf = self._predict()
            self._current_sign = sign
            self._current_conf = conf

            if sign and sign != self._last_committed:
                meaning = SIGN_MEANINGS.get(sign, sign.replace('_', ' ').title())
                self._current_meaning = meaning
                self._meaning_hist.append((sign, meaning))
                self._last_committed = sign
                print(f"  ▶ [{sign}]  {conf:.0%}  →  {meaning}")
            elif not sign:
                self._last_committed = None

            # Vẽ
            frame = self._draw_hands(frame, hand_result)
            frame = self._draw_face(frame, face_result, sign)
            frame = self._draw_hud(frame, sign, conf, self._current_meaning)

            cv2.imshow('VSL Combined Tester', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('c'), ord('C')):
                self._meaning_hist.clear()
                self._current_meaning = ""
                self._last_committed  = None
                print("  History cleared.")
            elif key == ord(' '):
                self.buffer.clear()
                self._sign_buf.clear()
                self._last_committed = None
                print("  Buffer reset.")

        cap.release()
        cv2.destroyAllWindows()
        print("\nPhiên kết thúc.")


def main():
    try:
        tester = CombinedTester()
        tester.run()
    except FileNotFoundError as e:
        print(f"\n✗ File không tồn tại: {e}")
        print("  Thứ tự chạy: 1) combined_data_collector.py → 2) combined_trainer.py → 3) combined_tester.py")
    except Exception as e:
        import traceback
        print(f"\n✗ Lỗi: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()