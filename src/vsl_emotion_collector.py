"""
VSL Data Collector - MediaPipe 0.10.32+ (Task API)
With Face Mesh Emotion Detection - Fixed & Improved
"""

import cv2
import numpy as np
import os
import json
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')


class FaceMeshEmotionDetector:
    """Detect emotions based on facial landmarks - Improved version"""

    def __init__(self):
        self.LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES     = [61, 291, 0, 17, 269, 405]
        self.LEFT_EYEBROW_INDICES  = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336]

        # Calibration buffers (running averages for neutral face)
        self._ear_history = []
        self._mar_history = []
        self._brow_history = []
        self._calibrated = False
        self._neutral_ear  = 0.25
        self._neutral_mar  = 0.10
        self._neutral_brow = 0.03
        self._calib_frames = 0
        self._calib_target = 60  # frames needed to calibrate

    # ------------------------------------------------------------------ helpers

    def _dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _ear(self, pts):
        """Eye Aspect Ratio"""
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _mar(self, pts):
        """Mouth Aspect Ratio"""
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _brow_pos(self, brow_pts, eye_pts):
        """Eyebrow position relative to eye (positive = raised)"""
        brow_y = np.mean([p[1] for p in brow_pts])
        eye_y  = np.mean([p[1] for p in eye_pts])
        return eye_y - brow_y

    # ------------------------------------------------------------------ calibrate

    def update_calibration(self, ear, mar, brow):
        """Auto-calibrate neutral face during first N frames"""
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
            self._calibrated = True
            print(f"[Calibration] EAR={self._neutral_ear:.3f}  "
                  f"MAR={self._neutral_mar:.3f}  BROW={self._neutral_brow:.3f}")

    # ------------------------------------------------------------------ detect

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

        # Auto-calibrate
        self.update_calibration(ear, mar, brow)

        # Relative deviations from neutral baseline
        d_ear  = ear  - self._neutral_ear    # positive = eyes wider
        d_mar  = mar  - self._neutral_mar    # positive = mouth opener
        d_brow = brow - self._neutral_brow   # positive = brows raised

        # ---- Decision tree (order matters) --------------------------------
        # HAPPY: mouth significantly open, brows NOT furrowed
        if d_mar > 0.12 and d_brow > -0.012:
            conf = min(d_mar * 500, 100)
            return "happy", conf

        # SURPRISE: eyes wide + brows raised + mouth open
        if d_ear > 0.05 and d_brow > 0.008 and d_mar > 0.08:
            conf = min((d_ear + d_brow + d_mar) * 200, 100)
            return "surprise", conf

        # WORRIED/ANXIOUS: brows raised slightly + eyes wide + mouth only slightly open
        if d_brow > 0.005 and d_ear > 0.02 and d_mar < 0.08:
            conf = min((d_brow + d_ear) * 600, 100)
            return "worried", conf

        # SAD: brows lowered + mouth only slightly open or closed
        if d_brow < -0.010 and d_mar < 0.08:
            conf = min(abs(d_brow) * 700, 100)
            return "sad", conf

        # ANGRY: brows strongly furrowed + eyes narrowed
        if d_brow < -0.012 and d_ear < -0.02:
            conf = min((abs(d_brow) + abs(d_ear)) * 400, 100)
            return "angry", conf

        # DISGUST: brows lowered + eyes slightly narrowed (less extreme than angry)
        if d_brow < -0.008 and -0.02 <= d_ear < 0.0:
            conf = min(abs(d_brow) * 500, 100)
            return "disgust", conf

        # NEUTRAL fallback
        return "neutral", 60.0

    @property
    def calibration_progress(self):
        if self._calibrated:
            return 100
        return int(self._calib_frames / self._calib_target * 100)


# ============================================================= Collector class

class VSLDataCollectorWithFaceMesh:
    EMOTION_COLORS = {
        'happy':   (0, 255, 0),
        'sad':     (255, 80, 80),
        'angry':   (0, 0, 255),
        'surprise':(255, 255, 0),
        'worried': (0, 165, 255),
        'disgust': (128, 0, 128),
        'neutral': (200, 200, 200),
    }
    EMOJIS = {
        'happy':   ':)',
        'sad':     ':(',
        'angry':   '>:(',
        'surprise':':O',
        'worried': ':S',
        'disgust': ':/',
        'neutral': ':|',
    }

    def __init__(self, output_dir=RAW_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("Initializing MediaPipe 0.10.32 Task API...")
        self._init_hand_detector()
        self._init_face_mesh()
        self.emotion_detector = FaceMeshEmotionDetector()

        self.metadata = {
            'signs': {},
            'total_samples': 0,
            'emotion_stats': {}
        }

        print("✓ MediaPipe initialized")
        print("✓ Face Mesh + Emotion detection enabled")

    # ---------------------------------------------------------------- init MP

    def _init_hand_detector(self):
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            import urllib.request
            url = ('https://storage.googleapis.com/mediapipe-models/'
                   'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
            urllib.request.urlretrieve(url, model_path)
            print("✓ Hand model downloaded")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

    def _init_face_mesh(self):
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            import urllib.request
            url = ('https://storage.googleapis.com/mediapipe-models/'
                   'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
            urllib.request.urlretrieve(url, model_path)
            print("✓ Face mesh downloaded")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

    # --------------------------------------------------------------- keypoints

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

    # ------------------------------------------------------------------ draw

    def draw_hand_landmarks(self, frame, detection_result):
        if not detection_result.hand_landmarks:
            return frame
        h, w, _ = frame.shape
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for hand_lm in detection_result.hand_landmarks:
            for lm in hand_lm:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            for a, b in CONNECTIONS:
                pa = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                pb = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
        return frame

    def draw_face_with_emotion(self, frame, face_result):
        if not face_result.face_landmarks:
            return frame, 0, None, 0.0

        h, w, _ = frame.shape
        face_count = len(face_result.face_landmarks)
        emotion, conf = self.emotion_detector.detect_emotion(face_result.face_landmarks)
        color = self.EMOTION_COLORS.get(emotion, (0, 255, 255))

        for face_lm in face_result.face_landmarks:
            KEY_IDX = [33, 133, 362, 263, 61, 291, 199]
            for idx in KEY_IDX:
                if idx < len(face_lm):
                    x = int(face_lm[idx].x * w)
                    y = int(face_lm[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            xs = [lm.x * w for lm in face_lm]
            ys = [lm.y * h for lm in face_lm]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{emotion.upper()} {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x_min, y_min - th - 10), (x_min + tw + 10, y_min), color, -1)
            cv2.putText(frame, text, (x_min + 5, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            emoji = self.EMOJIS.get(emotion, '')
            if emoji:
                cv2.putText(frame, emoji, (x_max + 10, y_min + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        return frame, face_count, emotion, conf

    def _draw_calibration_bar(self, frame, progress):
        h, w = frame.shape[:2]
        if progress >= 100:
            return
        bar_w, bar_h = 300, 18
        bx, by = w // 2 - bar_w // 2, h - 60
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * progress / 100)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (0, 200, 255), -1)
        cv2.putText(frame, f"Calibrating neutral face... {progress}%",
                    (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    # --------------------------------------------------------------- collect

    def collect_sign(self, sign_name, num_samples=30, sequence_length=30):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return 0

        sign_dir = os.path.join(self.output_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)

        sample_count = 0
        recording    = False
        sequence     = []
        emotion_log  = []

        print(f"\n{'='*60}")
        print(f"Collecting: {sign_name.upper()}")
        print(f"Target: {num_samples} samples  |  Sequence length: {sequence_length} frames")
        print(f"{'='*60}")
        print("Instructions:")
        print("  • Keep your face visible so neutral is calibrated first (~2 sec)")
        print("  • Press 'S' to START recording a sample")
        print("  • Perform the sign for 1 second")
        print("  • Press 'Q' to QUIT\n")

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_mesh.detect(mp_image)

            frame = self.draw_hand_landmarks(frame, hand_result)
            frame, face_count, emotion, emotion_conf = self.draw_face_with_emotion(frame, face_result)

            h, w = frame.shape[:2]

            # ---- Status panel ----
            cv2.rectangle(frame, (0, 0), (460, 270), (0, 0, 0), -1)

            status_text  = "RECORDING..." if recording else "Press 'S' to START"
            status_color = (0, 0, 255)    if recording else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv2.putText(frame, f"Sign: {sign_name}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            face_color = (0,255,0) if face_count == 1 else (0,165,255) if face_count > 1 else (0,0,255)
            cv2.putText(frame, f"Faces: {face_count}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Conf: {emotion_conf:.0f}%", (10, 235),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # ---- Progress bar while recording ----
            if recording:
                cv2.putText(frame, f"Frames: {len(sequence)}/{sequence_length}",
                            (10, h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                bw = 300
                bx, by = 10, h - 120
                progress = len(sequence) / sequence_length
                cv2.rectangle(frame, (bx, by), (bx + bw, by + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + 20), (0, 255, 255), -1)
                cv2.putText(frame, f"{int(progress*100)}%",
                            (bx + bw + 10, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # ---- Calibration bar ----
            calib = self.emotion_detector.calibration_progress
            self._draw_calibration_bar(frame, calib)

            # ---- Corner indicator ----
            if face_count == 0:
                cv2.putText(frame, "No face!", (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif face_count > 1:
                cv2.putText(frame, f"{face_count} faces!", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Face OK", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, "Press 'S' to start | 'Q' to quit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # ---- Capture ----
            if recording:
                kps = self.extract_keypoints(hand_result)
                kps = self.normalize_keypoints(kps)
                sequence.append(kps)
                if emotion:
                    emotion_log.append(emotion)

                if len(sequence) >= sequence_length:
                    sample_path = os.path.join(sign_dir, f'sample_{sample_count:03d}.npy')
                    np.save(sample_path, np.array(sequence))

                    if emotion_log:
                        dominant = max(set(emotion_log), key=emotion_log.count)
                        meta = {
                            'dominant_emotion': dominant,
                            'emotion_sequence': emotion_log,
                            'method': 'face_mesh_v2'
                        }
                        with open(sample_path.replace('.npy', '_emotion.json'), 'w') as f:
                            json.dump(meta, f)

                    sample_count += 1
                    sequence     = []
                    emotion_log  = []
                    recording    = False
                    print(f"  ✓ Saved sample {sample_count}/{num_samples}  (emotion: {emotion})")
                    time.sleep(0.3)

            cv2.imshow('VSL Data Collection + Face Mesh', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                if not recording:
                    recording   = True
                    sequence    = []
                    emotion_log = []
                    print(f"→ Recording sample {sample_count + 1}...")
            elif key in (ord('q'), ord('Q')):
                print("\n⊘ Stopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        self.metadata['signs'][sign_name] = {
            'num_samples': sample_count,
            'sequence_length': sequence_length
        }
        self.metadata['total_samples'] += sample_count
        print(f"\n✓ Completed: {sign_name} — {sample_count} samples")
        return sample_count

    def save_metadata(self):
        path = os.path.join(self.output_dir, 'metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Metadata saved: {path}")


# ================================================================= main

def main():
    collector = VSLDataCollectorWithFaceMesh()

    print("\n" + "=" * 60)
    print("VSL DATA COLLECTION WITH FACE MESH EMOTION (v2)")
    print("Supported emotions: happy, sad, angry, surprise, worried, disgust, neutral")
    print("=" * 60)

    signs = []
    while True:
        sign = input("\nNhập tên sign (Enter để kết thúc): ").strip()
        if sign == "":
            break
        signs.append(sign)

    if not signs:
        print("No signs specified. Exiting.")
        return

    print("\n" + "=" * 60)
    print(f"Signs to collect: {', '.join(signs)}")
    print("=" * 60)

    for sign in signs:
        resp = input(f"\nCollect '{sign}'? (y/n): ").strip().lower()
        if resp == 'y':
            collector.collect_sign(sign, num_samples=30)
        else:
            print(f"⊘ Skipped '{sign}'")

    collector.save_metadata()

    print("\n" + "=" * 60)
    print("✓ COLLECTION COMPLETE!")
    print(f"Total: {collector.metadata['total_samples']} samples")
    print("=" * 60)


if __name__ == '__main__':
    main()