"""
VSL Emotion Data Collector - Fixed Version
Thu thập dữ liệu facial landmarks để train emotion classifier
"""

import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'emotion_raw')

EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'worried', 'disgust', 'neutral']


class EmotionDataCollector:

    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX     = [61, 291, 0, 17, 269, 405]
    L_BROW_IDX    = [70, 63, 105, 66, 107]
    R_BROW_IDX    = [300, 293, 334, 296, 336]

    def __init__(self, output_dir=EMOTION_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        for emo in EMOTIONS:
            os.makedirs(os.path.join(output_dir, emo), exist_ok=True)
        self._init_face_mesh()
        print("FaceLandmarker initialized OK")

    def _find_or_download_model(self, filename, url):
        # Tim file o nhieu vi tri
        search_paths = [
            os.path.join(BASE_DIR, filename),
            os.path.join(BASE_DIR, '..', filename),
            os.path.join(os.getcwd(), filename),
        ]
        for path in search_paths:
            if os.path.exists(path):
                print(f"  Tim thay model: {os.path.abspath(path)}")
                return os.path.abspath(path)

        # Download neu khong tim thay
        save_path = os.path.join(BASE_DIR, filename)
        print(f"  Khong tim thay {filename}, dang download...")
        print(f"  URL: {url}")
        try:
            import urllib.request
            def progress(count, block_size, total_size):
                if total_size > 0:
                    pct = min(int(count * block_size * 100 / total_size), 100)
                    print(f"\r  Tien do: {pct}%", end='', flush=True)
            urllib.request.urlretrieve(url, save_path, reporthook=progress)
            print(f"\n  Download xong: {save_path}")
            return save_path
        except Exception as e:
            print(f"\n  LOI download: {e}")
            print(f"\n  Hay tai thu cong tai:")
            print(f"  {url}")
            print(f"  Va dat vao thu muc: {BASE_DIR}")
            raise FileNotFoundError(f"Khong tim thay va khong download duoc {filename}")

    def _init_face_mesh(self):
        print("\n[Init] Tim kiem face_landmarker.task...")
        model_path = self._find_or_download_model(
            'face_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        )
        print("[Init] Khoi tao FaceLandmarker...")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        print("[Init] OK")

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
        if not face_landmarks:
            return None
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
        eye_dist = self._dist(leye, reye)
        nose_chin = self._dist(nose, chin) / (eye_dist + 1e-6)
        return np.array([left_ear, right_ear, avg_ear, mar,
                         left_brow, right_brow, avg_brow,
                         nose_chin, eye_dist], dtype=np.float32)

    def collect_emotion(self, emotion: str, num_samples: int = 100):
        if emotion not in EMOTIONS:
            print(f"Emotion '{emotion}' khong hop le. Chon: {EMOTIONS}")
            return 0

        # Tim camera
        cap = None
        for idx in [1, 2]:
            test = cv2.VideoCapture(idx)
            if test.isOpened():
                ret, _ = test.read()
                if ret:
                    cap = test
                    print(f"  Dung camera index {idx}")
                    break
                test.release()
            else:
                test.release()

        if cap is None:
            print("Khong mo duoc camera!")
            return 0

        emo_dir = os.path.join(self.output_dir, emotion)
        existing = len([f for f in os.listdir(emo_dir) if f.endswith('.npy')])
        sample_count = existing
        auto_collect_count = 0
        auto_target = 0

        print(f"\n{'='*60}")
        print(f"Thu thap cam xuc: {emotion.upper()}")
        print(f"  Hien co: {existing} mau | Muc tieu them: {num_samples} mau")
        print(f"{'='*60}")
        print("Huong dan:")
        print("  Nhan 'S' de luu 1 mau")
        print("  Nhan 'A' de auto-collect 50 frames")
        print("  Nhan 'Q' de thoat\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Khong doc duoc frame!")
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self.face_mesh.detect(mp_image)
            features = self.extract_features(result.face_landmarks)

            h, w = frame.shape[:2]

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                xs = [l.x * w for l in lm]
                ys = [l.y * h for l in lm]
                cv2.rectangle(frame, (int(min(xs)), int(min(ys))),
                              (int(max(xs)), int(max(ys))), (0, 255, 0), 2)

            # Auto-collect
            if auto_collect_count < auto_target and features is not None:
                path = os.path.join(emo_dir, f'sample_{sample_count:04d}.npy')
                np.save(path, features)
                sample_count += 1
                auto_collect_count += 1
                time.sleep(0.05)
                if auto_collect_count >= auto_target:
                    print(f"  Auto-collected {auto_target} mau. Tong: {sample_count}")
                    auto_collect_count = 0
                    auto_target = 0

            # HUD
            cv2.rectangle(frame, (0, 0), (460, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"Emotion: {emotion.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count-existing}/{num_samples}  (Total:{sample_count})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            face_color = (0, 255, 0) if features is not None else (0, 0, 255)
            face_text  = "Face: OK" if features is not None else "Face: NOT DETECTED"
            cv2.putText(frame, face_text, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

            if features is not None:
                cv2.putText(frame,
                    f"EAR={features[2]:.3f} MAR={features[3]:.3f} BROW={features[6]:.3f}",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.putText(frame, "S:Save  A:Auto(50)  Q:Quit",
                        (w-260, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            if (sample_count - existing) >= num_samples:
                cv2.putText(frame, "TARGET REACHED! Nhan Q de tiep",
                            (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow(f'Emotion Collector - {emotion}', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                if features is not None:
                    path = os.path.join(emo_dir, f'sample_{sample_count:04d}.npy')
                    np.save(path, features)
                    sample_count += 1
                    print(f"  Luu mau {sample_count} ({emotion})")
                else:
                    print("  Khong phat hien khuon mat!")
            elif key in (ord('a'), ord('A')):
                auto_collect_count = 0
                auto_target = 50
                print("  Auto-collecting 50 frames...")
            elif key in (ord('q'), ord('Q')):
                break

        cap.release()
        cv2.destroyAllWindows()
        collected = sample_count - existing
        print(f"\nDa thu thap {collected} mau moi cho '{emotion}'. Tong: {sample_count}")
        return collected


def main():
    print("\n" + "="*60)
    print("EMOTION DATA COLLECTOR")
    print("="*60)

    try:
        collector = EmotionDataCollector()
    except Exception as e:
        print(f"\nLoi khoi tao: {e}")
        print("\nKiem tra:")
        print("  1. File 'face_landmarker.task' co trong thu muc src/ khong?")
        print("  2. Neu chua co, tai tai:")
        print("     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("  3. Dat file vao thu muc src/")
        input("\nNhan Enter de thoat...")
        return

    print("\n" + "="*60)
    print(f"Cam xuc ho tro: {', '.join(EMOTIONS)}")
    print("Can it nhat 100-200 mau moi cam xuc de train tot")
    print("="*60)

    while True:
        print(f"\nDanh sach: {EMOTIONS}")
        emo = input("Nhap ten cam xuc (Enter de thoat): ").strip().lower()
        if emo == "":
            break
        if emo not in EMOTIONS:
            print(f"Khong hop le. Chon tu: {EMOTIONS}")
            continue
        try:
            n_str = input(f"So mau cho '{emo}' (mac dinh 100): ").strip()
            n = int(n_str) if n_str else 100
        except ValueError:
            n = 100
        collector.collect_emotion(emo, num_samples=n)

    print("\nHoan thanh!")
    print(f"Data luu tai: {os.path.abspath(EMOTION_DATA_DIR)}")
    input("Nhan Enter de thoat...")


if __name__ == '__main__':
    main()