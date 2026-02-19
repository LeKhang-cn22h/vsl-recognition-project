"""
VSL Real-time Tester - MediaPipe 0.10.32 Task API
With Face Mesh Emotion Detection
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================= Emotion Detector

class FaceMeshEmotionDetector:
    """Detect emotions based on facial landmarks"""

    def __init__(self):
        self.LEFT_EYE_INDICES       = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES      = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES          = [61, 291, 0, 17, 269, 405]
        self.LEFT_EYEBROW_INDICES   = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW_INDICES  = [300, 293, 334, 296, 336]

        self._ear_history   = []
        self._mar_history   = []
        self._brow_history  = []
        self._calibrated    = False
        self._neutral_ear   = 0.25
        self._neutral_mar   = 0.10
        self._neutral_brow  = 0.03
        self._calib_frames  = 0
        self._calib_target  = 60

    def _dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _ear(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _mar(self, pts):
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _brow_pos(self, brow_pts, eye_pts):
        brow_y = np.mean([p[1] for p in brow_pts])
        eye_y  = np.mean([p[1] for p in eye_pts])
        return eye_y - brow_y

    def update_calibration(self, ear, mar, brow):
        if self._calibrated:
            return
        self._ear_history.append(ear)
        self._mar_history.append(mar)
        self._brow_history.append(brow)
        self._calib_frames += 1
        if self._calib_frames >= self._calib_target:
            self._neutral_ear  = float(np.percentile(self._ear_history, 50))
            self._neutral_mar  = float(np.percentile(self._mar_history, 50))
            self._neutral_brow = float(np.percentile(self._brow_history, 50))
            self._calibrated   = True
            print(f"[Calibration done] EAR={self._neutral_ear:.3f}  "
                  f"MAR={self._neutral_mar:.3f}  BROW={self._neutral_brow:.3f}")

    def detect_emotion(self, face_landmarks):
        if not face_landmarks:
            return "neutral", 0.0

        lm = face_landmarks[0]
        xy = lambda i: (lm[i].x, lm[i].y)

        left_eye  = [xy(i) for i in self.LEFT_EYE_INDICES]
        right_eye = [xy(i) for i in self.RIGHT_EYE_INDICES]
        mouth     = [xy(i) for i in self.MOUTH_INDICES]
        l_brow    = [xy(i) for i in self.LEFT_EYEBROW_INDICES]
        r_brow    = [xy(i) for i in self.RIGHT_EYEBROW_INDICES]

        ear  = (self._ear(left_eye) + self._ear(right_eye)) / 2.0
        mar  = self._mar(mouth)
        brow = (self._brow_pos(l_brow, left_eye) + self._brow_pos(r_brow, right_eye)) / 2.0

        self.update_calibration(ear, mar, brow)

        d_ear  = ear  - self._neutral_ear
        d_mar  = mar  - self._neutral_mar
        d_brow = brow - self._neutral_brow

        if d_mar > 0.12 and d_brow > -0.012:
            return "happy",    min(d_mar * 500, 100)
        if d_ear > 0.05 and d_brow > 0.008 and d_mar > 0.08:
            return "surprise", min((d_ear + d_brow + d_mar) * 200, 100)
        if d_brow > 0.005 and d_ear > 0.02 and d_mar < 0.08:
            return "worried",  min((d_brow + d_ear) * 600, 100)
        if d_brow < -0.010 and d_mar < 0.08:
            return "sad",      min(abs(d_brow) * 700, 100)
        if d_brow < -0.012 and d_ear < -0.02:
            return "angry",    min((abs(d_brow) + abs(d_ear)) * 400, 100)
        if d_brow < -0.008 and -0.02 <= d_ear < 0.0:
            return "disgust",  min(abs(d_brow) * 500, 100)
        return "neutral", 60.0

    @property
    def calibration_progress(self):
        if self._calibrated:
            return 100
        return int(self._calib_frames / self._calib_target * 100)


# ============================================================= VSL Tester

class VSLTester:
    EMOTION_COLORS = {
        'happy':    (0, 255, 0),
        'sad':      (255, 80, 80),
        'angry':    (0, 0, 255),
        'surprise': (255, 255, 0),
        'worried':  (0, 165, 255),
        'disgust':  (128, 0, 128),
        'neutral':  (200, 200, 200),
    }
    EMOJIS = {
        'happy':    ':)',
        'sad':      ':(',
        'angry':    '>:(',
        'surprise': ':O',
        'worried':  ':S',
        'disgust':  ':/',
        'neutral':  ':|',
    }

    def __init__(self):
        # ---- Load sign recognition model ----
        model_paths   = ['../models/vsl_model.h5',    'models/vsl_model.h5']
        encoder_paths = ['../models/label_encoder.npy', 'models/label_encoder.npy']

        # Also search for best_model_* files
        for d in ['../models', 'models']:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.startswith('best_model_') and f.endswith('.h5'):
                        model_paths.insert(0, os.path.join(d, f))

        model_path   = next((p for p in model_paths   if os.path.exists(p)), None)
        encoder_path = next((p for p in encoder_paths if os.path.exists(p)), None)

        if model_path is None or encoder_path is None:
            print("❌ Model not found! Train a model first: python src/train_model.py")
            exit(1)

        print(f"Loading sign model: {model_path}")
        self.model  = tf.keras.models.load_model(model_path)
        self.labels = np.load(encoder_path, allow_pickle=True)
        print(f"✓ Sign labels: {self.labels}")

        # ---- MediaPipe Hand Landmarker ----
        hand_model = 'hand_landmarker.task'
        if not os.path.exists(hand_model):
            print("Downloading hand landmarker model...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/'
                'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                hand_model
            )
            print("✓ Hand model downloaded")

        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # ---- MediaPipe Face Landmarker ----
        face_model = 'face_landmarker.task'
        if not os.path.exists(face_model):
            print("Downloading face landmarker model...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                face_model
            )
            print("✓ Face model downloaded")

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

        # ---- Emotion detector ----
        from ml_emotion_detector import MLEmotionDetector
        self.emotion_detector = MLEmotionDetector()

        # ---- Prediction buffer (30 frames) ----
        self.buffer = deque(maxlen=30)

        # ---- Smoothing: keep last N emotion predictions ----
        self._emotion_history = deque(maxlen=10)

        print("✓ All models loaded. Ready!")

    # ---------------------------------------------------------- helpers

    def extract_keypoints(self, detection_result):
        keypoints = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for lm in hand_landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints = [0] * 63
        while len(keypoints) < 126:
            keypoints.extend([0] * 63)
        return keypoints[:126]

    def normalize_keypoints(self, keypoints):
        kps = np.array(keypoints).reshape(-1, 3)
        for hand_idx in range(2):
            s, e = hand_idx * 21, hand_idx * 21 + 21
            hand_kps = kps[s:e]
            if np.any(hand_kps != 0):
                hand_kps -= hand_kps[0].copy()
                kps[s:e] = hand_kps
        return kps.flatten().tolist()

    def predict_sign(self):
        if len(self.buffer) < 30:
            return None, 0.0
        seq  = np.expand_dims(np.array(list(self.buffer)), axis=0)
        pred = self.model.predict(seq, verbose=0)[0]
        conf = float(np.max(pred))
        idx  = int(np.argmax(pred))
        if conf < 0.6:
            return None, conf
        return self.labels[idx], conf

    def smooth_emotion(self, emotion):
        """Return majority vote over recent emotion history."""
        self._emotion_history.append(emotion)
        return max(set(self._emotion_history), key=list(self._emotion_history).count)

    # ---------------------------------------------------------- draw

    def draw_hands(self, frame, detection_result):
        if not detection_result.hand_landmarks:
            return frame
        h, w, _ = frame.shape
        CONNS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for hand_lm in detection_result.hand_landmarks:
            for lm in hand_lm:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            for a, b in CONNS:
                pa = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                pb = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
        return frame

    def draw_face_emotion(self, frame, face_result):
        """Draw face bounding box + emotion label. Returns (frame, emotion, conf)."""
        if not face_result.face_landmarks:
            return frame, "neutral", 0.0

        h, w, _ = frame.shape
        raw_emotion, conf = self.emotion_detector.detect_emotion(face_result.face_landmarks)
        emotion = self.smooth_emotion(raw_emotion)
        color   = self.EMOTION_COLORS.get(emotion, (0, 255, 255))

        for face_lm in face_result.face_landmarks:
            xs = [lm.x * w for lm in face_lm]
            ys = [lm.y * h for lm in face_lm]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))

            # Draw key points
            for idx in [33, 133, 362, 263, 61, 291, 199]:
                if idx < len(face_lm):
                    cv2.circle(frame,
                               (int(face_lm[idx].x * w), int(face_lm[idx].y * h)),
                               2, (0, 255, 255), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{emotion.upper()}  {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            emoji = self.EMOJIS.get(emotion, '')
            if emoji:
                cv2.putText(frame, emoji, (x2 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

        return frame, emotion, conf

    def _draw_calibration_bar(self, frame):
        progress = self.emotion_detector.calibration_progress
        if progress >= 100:
            return
        h, w = frame.shape[:2]
        bw, bh = 300, 18
        bx, by = w // 2 - bw // 2, h - 55
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * progress / 100), by + bh), (0, 200, 255), -1)
        cv2.putText(frame, f"Calibrating face... {progress}%",
                    (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # ---------------------------------------------------------- run

    def run(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return

        print("\n" + "=" * 55)
        print("  VSL REAL-TIME RECOGNITION  |  Hand + Face Emotion")
        print("=" * 55)
        print("  Keep face visible for ~2s to calibrate neutral.")
        print("  Press 'Q' to quit.")
        print("=" * 55 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_mesh.detect(mp_image)

            # Draw landmarks
            frame = self.draw_hands(frame, hand_result)
            frame, emotion, emotion_conf = self.draw_face_emotion(frame, face_result)

            # Sign prediction
            kps = self.normalize_keypoints(self.extract_keypoints(hand_result))
            self.buffer.append(kps)
            sign, sign_conf = self.predict_sign()

            h, w = frame.shape[:2]

            # ---- HUD: sign recognition panel ----
            cv2.rectangle(frame, (0, 0), (420, 130), (0, 0, 0), -1)

            if sign:
                cv2.putText(frame, sign.upper(), (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {sign_conf:.0%}", (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Recognizing...", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 180), 2)

            # ---- Emotion summary (bottom-left) ----
            emotion_color = self.EMOTION_COLORS.get(emotion, (200, 200, 200))
            cv2.rectangle(frame, (0, h - 90), (340, h), (0, 0, 0), -1)
            cv2.putText(frame, f"Emotion: {emotion.upper()}  ({emotion_conf:.0f}%)",
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            # Buffer progress bar
            buf_progress = len(self.buffer) / 30
            bx, by, bw = 10, h - 30, 200
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 15), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * buf_progress), by + 15),
                          (0, 200, 100), -1)
            cv2.putText(frame, f"Buffer {len(self.buffer)}/30",
                        (bx + bw + 8, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Calibration bar
            self._draw_calibration_bar(frame)

            # Quit hint
            cv2.putText(frame, "Q: quit", (w - 100, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow('VSL Real-time Recognition', frame)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Session ended.")


def main():
    tester = VSLTester()
    tester.run()


if __name__ == '__main__':
    main()