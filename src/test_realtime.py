"""
VSL Real-time Tester
Nhận diện ký hiệu tay + biểu cảm mặt → in ra ý nghĩa ngôn ngữ ký hiệu
Không cần train emotion riêng, dùng rule-based từ facial landmarks
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ================================================================= MEANING MAP
# Ghép (sign, emotion) → câu có nghĩa
# Thêm/sửa theo từ điển ký hiệu của bạn

SIGN_EMOTION_MEANING = {
    # --- HELLO ---
    ('hello', 'happy'):    "Xin chào! (vui vẻ)",
    ('hello', 'sad'):      "Xin chào... (buồn bã)",
    ('hello', 'surprised'):"Ồ, xin chào!",
    ('hello', 'angry'):    "Xin chào. (không vui)",
    ('hello', 'neutral'):  "Xin chào.",
    ('hello', 'worried'):  "Xin chào... (lo lắng)",
    ('hello', 'disgust'):  "Xin chào. (khó chịu)",

    # --- GOODBYE ---
    ('goodbye', 'happy'):  "Tạm biệt nhé! (vui vẻ)",
    ('goodbye', 'sad'):    "Tạm biệt... (tiếc nuối)",
    ('goodbye', 'neutral'):"Tạm biệt.",
    ('goodbye', 'angry'):  "Tạm biệt! (tức giận)",
    ('goodbye', 'worried'):"Tạm biệt, hẹn gặp lại.",

    # --- YES ---
    ('yes', 'happy'):      "Có! Đồng ý! (hào hứng)",
    ('yes', 'sad'):        "Có... (miễn cưỡng)",
    ('yes', 'angry'):      "Có! (dứt khoát)",
    ('yes', 'neutral'):    "Có.",
    ('yes', 'surprised'):  "Ừ thật sao?!",

    # --- NO ---
    ('no', 'angry'):       "Không! (kiên quyết)",
    ('no', 'sad'):         "Không... (thất vọng)",
    ('no', 'neutral'):     "Không.",
    ('no', 'worried'):     "Không, tôi lo lắng.",
    ('no', 'disgust'):     "Không! (phản đối)",

    # --- THANK YOU ---
    ('thank_you', 'happy'):   "Cảm ơn bạn nhiều lắm!",
    ('thank_you', 'sad'):     "Cảm ơn... (xúc động)",
    ('thank_you', 'neutral'): "Cảm ơn.",
    ('thank_you', 'surprised'):"Ồ, cảm ơn bạn!",

    # --- SORRY ---
    ('sorry', 'sad'):      "Xin lỗi, tôi rất tiếc.",
    ('sorry', 'worried'):  "Xin lỗi, tôi lo lắng.",
    ('sorry', 'neutral'):  "Xin lỗi.",
    ('sorry', 'angry'):    "Xin lỗi! (chân thành)",

    # --- HELP ---
    ('help', 'worried'):   "Giúp tôi với, tôi lo lắng!",
    ('help', 'sad'):       "Tôi cần giúp đỡ...",
    ('help', 'angry'):     "Hãy giúp tôi ngay!",
    ('help', 'neutral'):   "Tôi cần giúp đỡ.",

    # --- GOOD ---
    ('good', 'happy'):     "Tốt lắm! Tuyệt vời!",
    ('good', 'neutral'):   "Tốt.",
    ('good', 'surprised'): "Ồ, tốt thật!",

    # --- BAD ---
    ('bad', 'angry'):      "Tệ! Tôi không hài lòng.",
    ('bad', 'sad'):        "Tệ quá...",
    ('bad', 'neutral'):    "Không tốt.",
    ('bad', 'disgust'):    "Tệ, tôi ghét điều này.",

    # --- I LOVE YOU ---
    ('i_love_you', 'happy'):   "Tôi yêu bạn! ❤️",
    ('i_love_you', 'sad'):     "Tôi yêu bạn... (nhớ nhung)",
    ('i_love_you', 'neutral'): "Tôi yêu bạn.",
}

# Fallback: chỉ dùng sign nếu không có emotion match
SIGN_DEFAULT_MEANING = {
    'hello':      "Xin chào",
    'goodbye':    "Tạm biệt",
    'yes':        "Có / Đồng ý",
    'no':         "Không",
    'thank_you':  "Cảm ơn",
    'sorry':      "Xin lỗi",
    'help':       "Giúp đỡ",
    'good':       "Tốt",
    'bad':        "Tệ / Xấu",
    'i_love_you': "Tôi yêu bạn",
    'please':     "Làm ơn",
    'water':      "Nước",
    'eat':        "Ăn",
    'more':       "Thêm",
    'home':       "Nhà",
    'friend':     "Bạn bè",
    'family':     "Gia đình",
    'doctor':     "Bác sĩ",
    'pain':       "Đau",
    'tired':      "Mệt mỏi",
}


def get_meaning(sign: str, emotion: str) -> str:
    """Tra từ điển, fallback về sign-only nếu không có cặp (sign, emotion)"""
    key = (sign.lower(), emotion.lower())
    if key in SIGN_EMOTION_MEANING:
        return SIGN_EMOTION_MEANING[key]
    # Fallback: sign default + gắn thêm emotion
    sign_meaning = SIGN_DEFAULT_MEANING.get(sign.lower(), sign.upper())
    emotion_map  = {
        'happy':    '😊',
        'sad':      '😢',
        'angry':    '😠',
        'surprised': '😮',
        'worried':  '😟',
        'disgust':  '😒',
        'neutral':  '',
    }
    emoji = emotion_map.get(emotion, '')
    return f"{sign_meaning} {emoji}".strip()


# ================================================================= EMOTION DETECTOR (Rule-based)

class FaceEmotionDetector:
    """
    Rule-based emotion từ facial landmarks.
    Tự calibrate neutral trong 60 frame đầu.
    """

    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH     = [61, 291, 0, 17, 269, 405]
    L_BROW    = [70, 63, 105, 66, 107]
    R_BROW    = [300, 293, 334, 296, 336]

    def __init__(self):
        self._ear_hist  = []
        self._mar_hist  = []
        self._brow_hist = []
        self._calibrated   = False
        self._calib_frames = 0
        self._calib_target = 60
        self._n_ear  = 0.25
        self._n_mar  = 0.10
        self._n_brow = 0.03

    def _dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _ear(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2*h) if h > 0 else 0

    def _mar(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1+v2)/(2*h) if h > 0 else 0

    def _brow(self, brow_pts, eye_pts):
        return np.mean([p[1] for p in eye_pts]) - np.mean([p[1] for p in brow_pts])

    def detect(self, face_landmarks):
        """Trả về (emotion_str, confidence_0_to_100)"""
        if not face_landmarks:
            return "neutral", 0

        lm = face_landmarks[0]
        xy = lambda i: (lm[i].x, lm[i].y)

        le = [xy(i) for i in self.LEFT_EYE]
        re = [xy(i) for i in self.RIGHT_EYE]
        mo = [xy(i) for i in self.MOUTH]
        lb = [xy(i) for i in self.L_BROW]
        rb = [xy(i) for i in self.R_BROW]

        ear  = (self._ear(le) + self._ear(re)) / 2
        mar  = self._mar(mo)
        brow = (self._brow(lb, le) + self._brow(rb, re)) / 2

        # Calibration
        if not self._calibrated:
            self._ear_hist.append(ear)
            self._mar_hist.append(mar)
            self._brow_hist.append(brow)
            self._calib_frames += 1
            if self._calib_frames >= self._calib_target:
                self._n_ear  = float(np.percentile(self._ear_hist,  50))
                self._n_mar  = float(np.percentile(self._mar_hist,  50))
                self._n_brow = float(np.percentile(self._brow_hist, 50))
                self._calibrated = True

        de = ear  - self._n_ear
        dm = mar  - self._n_mar
        db = brow - self._n_brow

        # Classify (thứ tự quan trọng: đặc biệt → tổng quát)
        if de > 0.05 and db > 0.008 and dm > 0.08:
            return "surprised", min((de+db+dm)*200, 100)
        if dm > 0.12 and db > -0.012:
            return "happy",    min(dm*500, 100)
        if db > 0.005 and de > 0.02 and dm < 0.08:
            return "worried",  min((db+de)*600, 100)
        if db < -0.012 and de < -0.02:
            return "angry",    min((abs(db)+abs(de))*400, 100)
        if db < -0.008 and -0.02 <= de < 0:
            return "disgust",  min(abs(db)*500, 100)
        if db < -0.010 and dm < 0.08:
            return "sad",      min(abs(db)*700, 100)
        return "neutral", 60

    @property
    def calibration_progress(self):
        if self._calibrated:
            return 100
        return int(self._calib_frames / self._calib_target * 100)


# ================================================================= VSL TESTER

class VSLTester:

    EMOTION_COLORS = {
        'happy':     (0, 255, 0),
        'sad':       (255, 80, 80),
        'angry':     (0, 0, 255),
        'surprised': (255, 255, 0),
        'worried':   (0, 165, 255),
        'disgust':   (128, 0, 128),
        'neutral':   (200, 200, 200),
    }

    def __init__(self):
        # ---- Sign model ----
        model_paths   = ['../models/vsl_model.h5',    'models/vsl_model.h5']
        encoder_paths = ['../models/label_encoder.npy', 'models/label_encoder.npy']
        for d in ['../models', 'models']:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.startswith('best_model_') and f.endswith('.h5'):
                        model_paths.insert(0, os.path.join(d, f))

        model_path   = next((p for p in model_paths   if os.path.exists(p)), None)
        encoder_path = next((p for p in encoder_paths if os.path.exists(p)), None)
        if model_path is None or encoder_path is None:
            print("Model not found! Run: python src/train_model.py")
            exit(1)

        print(f"Loading sign model: {model_path}")
        self.model  = tf.keras.models.load_model(model_path)
        self.labels = np.load(encoder_path, allow_pickle=True)
        print(f"Sign labels: {self.labels}")

        # ---- MediaPipe Hand ----
        hand_model = 'hand_landmarker.task'
        if not os.path.exists(hand_model):
            self._download(hand_model,
                'https://storage.googleapis.com/mediapipe-models/'
                'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # ---- MediaPipe Face ----
        face_model = 'face_landmarker.task'
        if not os.path.exists(face_model):
            self._download(face_model,
                'https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
        self.face_mesh = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_model),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # ---- Rule-based emotion detector ----
        self.emotion_detector = FaceEmotionDetector()

        # ---- Buffers ----
        self.hand_buffer     = deque(maxlen=30)
        self._emotion_buf    = deque(maxlen=10)   # smooth emotion
        self._sign_buf       = deque(maxlen=5)    # smooth sign
        self._meaning_history = deque(maxlen=3)   # last 3 meanings hiển thị

        print("All models loaded. Ready!\n")

    def _download(self, filename, url):
        import urllib.request
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Done: {filename}")

    # -------------------------------------------------------- keypoints

    def _extract_keypoints(self, result):
        kps = []
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                for lm in hand_lm:
                    kps.extend([lm.x, lm.y, lm.z])
        else:
            kps = [0]*63
        while len(kps) < 126:
            kps.extend([0]*63)
        return kps[:126]

    def _normalize_keypoints(self, kps):
        arr = np.array(kps).reshape(-1, 3)
        for i in range(2):
            s, e = i*21, i*21+21
            h = arr[s:e]
            if np.any(h != 0):
                h -= h[0].copy()
                arr[s:e] = h
        return arr.flatten().tolist()

    # -------------------------------------------------------- prediction

    def _predict_sign(self):
        if len(self.hand_buffer) < 30:
            return None, 0.0
        seq  = np.expand_dims(np.array(list(self.hand_buffer)), axis=0)
        pred = self.model.predict(seq, verbose=0)[0]
        conf = float(np.max(pred))
        idx  = int(np.argmax(pred))
        if conf < 0.6:
            return None, conf
        raw_sign = self.labels[idx]
        # Smooth sign
        self._sign_buf.append(raw_sign)
        sign = max(set(self._sign_buf), key=list(self._sign_buf).count)
        return sign, conf

    def _smooth_emotion(self, emotion):
        self._emotion_buf.append(emotion)
        return max(set(self._emotion_buf), key=list(self._emotion_buf).count)

    # -------------------------------------------------------- draw

    HAND_CONNS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    def _draw_hands(self, frame, result):
        if not result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lm in result.hand_landmarks:
            for lm in hand_lm:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,0), -1)
            for a, b in self.HAND_CONNS:
                pa = (int(hand_lm[a].x*w), int(hand_lm[a].y*h))
                pb = (int(hand_lm[b].x*w), int(hand_lm[b].y*h))
                cv2.line(frame, pa, pb, (255,255,255), 2)
        return frame

    def _draw_face(self, frame, face_result, emotion, conf):
        if not face_result.face_landmarks:
            return frame
        h, w = frame.shape[:2]
        color = self.EMOTION_COLORS.get(emotion, (200,200,200))
        for face_lm in face_result.face_landmarks:
            xs = [lm.x*w for lm in face_lm]
            ys = [lm.y*h for lm in face_lm]
            x1,x2 = int(min(xs)), int(max(xs))
            y1,y2 = int(min(ys)), int(max(ys))
            # Key points
            for idx in [33,133,362,263,61,291,199]:
                if idx < len(face_lm):
                    cv2.circle(frame,
                               (int(face_lm[idx].x*w), int(face_lm[idx].y*h)),
                               2, (0,255,255), -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{emotion.upper()} {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        return frame

    def _draw_hud(self, frame, sign, sign_conf, emotion, emotion_conf, meaning):
        h, w = frame.shape[:2]
        emotion_color = self.EMOTION_COLORS.get(emotion, (200,200,200))

        # ---- Top panel: sign recognition ----
        cv2.rectangle(frame, (0,0), (w, 130), (0,0,0), -1)

        if sign:
            cv2.putText(frame, sign.upper(), (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3)
            cv2.putText(frame, f"Confidence: {sign_conf:.0%}",
                        (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "Recognizing...", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (150,150,150), 2)

        # ---- Emotion strip (right of sign panel) ----
        cv2.putText(frame, f"{emotion.upper()} ({emotion_conf:.0f}%)",
                    (w-240, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)

        # ---- MEANING BOX (giữa màn hình, nổi bật) ----
        if meaning:
            # Background box
            box_y = h // 2 - 40
            (tw, th), _ = cv2.getTextSize(meaning, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            bx = w//2 - tw//2 - 15
            by = box_y - th - 10
            cv2.rectangle(frame, (bx, by), (bx+tw+30, box_y+15), (0,0,0), -1)
            cv2.rectangle(frame, (bx, by), (bx+tw+30, box_y+15), (0,255,255), 2)
            cv2.putText(frame, meaning,
                        (w//2 - tw//2, box_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # ---- Meaning history (bottom) ----
        cv2.rectangle(frame, (0, h-110), (w, h), (0,0,0), -1)
        cv2.putText(frame, "Lich su:", (10, h-90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)
        for i, past in enumerate(reversed(list(self._meaning_history))):
            alpha = 255 - i*60
            color = (alpha, alpha, alpha)
            cv2.putText(frame, past, (10, h-70 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ---- Buffer bar ----
        buf = len(self.hand_buffer)
        bx, by2, bw2 = w-230, h-30, 180
        cv2.rectangle(frame, (bx, by2), (bx+bw2, by2+12), (50,50,50), -1)
        cv2.rectangle(frame, (bx, by2), (bx+int(bw2*buf/30), by2+12), (0,200,100), -1)
        cv2.putText(frame, f"Buf {buf}/30",
                    (bx+bw2+5, by2+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        # ---- Calibration bar ----
        prog = self.emotion_detector.calibration_progress
        if prog < 100:
            cw = 200
            cx = w//2 - cw//2
            cy = h - 55
            cv2.rectangle(frame, (cx,cy), (cx+cw, cy+14), (50,50,50), -1)
            cv2.rectangle(frame, (cx,cy), (cx+int(cw*prog/100), cy+14), (0,200,255), -1)
            cv2.putText(frame, f"Calibrating face {prog}%",
                        (cx, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)

        # Quit hint
        cv2.putText(frame, "Q: quit  C: clear history",
                    (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1)

        return frame

    # -------------------------------------------------------- run

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera!")
            return

        print("="*55)
        print("  VSL REAL-TIME  |  Ký hiệu tay + Biểu cảm mặt")
        print("="*55)
        print("  Giữ biểu cảm mặt và làm ký hiệu tay trong ~2 giây.")
        print("  Hệ thống sẽ hiện ý nghĩa ngay khi nhận ra.")
        print("="*55 + "\n")

        current_meaning = ""
        last_sign       = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_mesh.detect(mp_image)

            # --- Emotion ---
            raw_emotion, emotion_conf = self.emotion_detector.detect(face_result.face_landmarks)
            emotion = self._smooth_emotion(raw_emotion)

            # --- Sign ---
            kps = self._normalize_keypoints(self._extract_keypoints(hand_result))
            self.hand_buffer.append(kps)
            sign, sign_conf = self._predict_sign()

            # --- Ghép meaning ---
            if sign and sign != last_sign:
                current_meaning = get_meaning(sign, emotion)
                # Thêm vào history nếu khác entry cuối
                if not self._meaning_history or self._meaning_history[-1] != current_meaning:
                    self._meaning_history.append(current_meaning)
                last_sign = sign
                print(f"  [{sign.upper()} + {emotion}] → {current_meaning}")
            elif not sign:
                last_sign = None  # reset để nhận ký hiệu mới

            # --- Draw ---
            frame = self._draw_hands(frame, hand_result)
            frame = self._draw_face(frame, face_result, emotion, emotion_conf)
            frame = self._draw_hud(frame, sign, sign_conf,
                                   emotion, emotion_conf, current_meaning)

            cv2.imshow('VSL Real-time', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('c'), ord('C')):
                self._meaning_history.clear()
                current_meaning = ""
                last_sign = None
                print("  History cleared.")

        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended.")


def main():
    tester = VSLTester()
    tester.run()


if __name__ == '__main__':
    main()