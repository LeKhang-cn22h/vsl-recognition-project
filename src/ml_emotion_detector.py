import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

class MLEmotionDetector:
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX     = [61, 291, 0, 17, 269, 405]
    L_BROW_IDX    = [70, 63, 105, 66, 107]
    R_BROW_IDX    = [300, 293, 334, 296, 336]

    def __init__(self):
        model_path   = os.path.join(MODEL_DIR, 'emotion_classifier.pkl')
        encoder_path = os.path.join(MODEL_DIR, 'emotion_label_encoder.pkl')
        self.pipeline = joblib.load(model_path)
        self.encoder  = joblib.load(encoder_path)
        print(f"✓ Loaded ML emotion model: {model_path}")

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

    def extract_features(self, face_landmarks):
        lm = face_landmarks[0]
        xy = lambda i: (lm[i].x, lm[i].y)
        le = [xy(i) for i in self.LEFT_EYE_IDX]
        re = [xy(i) for i in self.RIGHT_EYE_IDX]
        mo = [xy(i) for i in self.MOUTH_IDX]
        lb = [xy(i) for i in self.L_BROW_IDX]
        rb = [xy(i) for i in self.R_BROW_IDX]
        left_ear  = self._ear(le)
        right_ear = self._ear(re)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = self._mar(mo)
        left_brow  = self._brow_pos(lb, le)
        right_brow = self._brow_pos(rb, re)
        avg_brow   = (left_brow + right_brow) / 2.0
        nose = xy(1); chin = xy(152)
        leye = xy(33); reye = xy(263)
        eye_dist  = self._dist(leye, reye)
        nose_chin = self._dist(nose, chin) / (eye_dist + 1e-6)
        return np.array([left_ear, right_ear, avg_ear, mar,
                         left_brow, right_brow, avg_brow,
                         nose_chin, eye_dist], dtype=np.float32)

    def detect_emotion(self, face_landmarks):
        if not face_landmarks:
            return "neutral", 0.0
        features = self.extract_features(face_landmarks).reshape(1, -1)
        proba    = self.pipeline.predict_proba(features)[0]
        idx      = int(np.argmax(proba))
        conf     = float(proba[idx]) * 100
        emotion  = self.encoder.classes_[idx]
        return emotion, conf

    # Để tương thích với calibration_progress trong VSLTester
    @property
    def calibration_progress(self):
        return 100