"""
VSL Real-time Tester - GCN Version

FIXES (accuracy):
- [BUG FIX] normalize_keypoints: Vectorize bằng numpy thay vì Python for loop
- [BUG FIX] motion_variance: Dùng index đúng trên data ĐÃ normalize (75 điểm tay)
            Trước đây dùng index 1533:1659 trên 1659-dim nhưng buffer lưu 1659-dim normalized,
            vẫn đúng index nhưng threshold cần điều chỉnh vì data đã scale theo vai

FIXES (speed):
- [SPEED]   Bỏ Face Detector — GCN chỉ dùng Pose + Hands, face tốn ~30% thời gian vô ích
- [SPEED]   model(input, training=False) thay vì model.predict() — nhanh hơn 3-5x với batch=1
- [SPEED]   Vectorize normalize_keypoints bằng numpy — nhanh hơn ~10x so với for loop
- [SPEED]   Tăng PREDICT_EVERY lên 20 để giảm tải, kết quả vẫn hiển thị mượt
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VSLGCNTester:
    PREDICT_EVERY    = 20     # Predict mỗi 20 frame (~1.5 lần/giây ở 30fps)
    MOTION_THRESHOLD = 0.001  # Threshold sau khi normalize theo vai (scale khác raw)
    CONF_THRESHOLD   = 0.5

    def __init__(self):
        self._load_resources()

        print("Initializing MediaPipe detectors...")
        self._init_detectors()

        self.buffer      = deque(maxlen=30)
        self.frame_count = 0
        self.last_sign   = ""
        self.last_conf   = 0.0
        self.in_motion   = False

        print("✓ Ready!")

    # ==========================================
    # NORMALIZE — vectorized, giống hệt collect
    # ==========================================
    def normalize_keypoints(self, keypoints):
        """
        Normalize theo midpoint 2 vai.
        Vectorized bằng numpy — nhanh hơn ~10x so với Python for loop.
        """
        kps = np.array(keypoints, dtype=np.float32).reshape(-1, 3)  # (553, 3)

        left_shoulder  = kps[11].copy()
        right_shoulder = kps[12].copy()

        if np.any(left_shoulder != 0) and np.any(right_shoulder != 0):
            center        = (left_shoulder + right_shoulder) / 2.0
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

            if shoulder_dist > 1e-6:
                # Vectorized: tìm mask các điểm detect được, trừ center, chia dist
                detected_mask        = np.any(kps != 0, axis=1)          # (553,) bool
                kps[detected_mask]   = (kps[detected_mask] - center) / shoulder_dist

        return kps.flatten()

    def _load_resources(self):
        model_path = '../models/best_gcn_model.h5'
        if not os.path.exists(model_path):
            model_path = 'models/best_gcn_model.h5'
        if not os.path.exists(model_path):
            print("❌ Model not found! Please run train_gcn.py first.")
            exit(1)

        encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder_gcn.npy')
        if not os.path.exists(encoder_path):
            print(f"❌ Label encoder not found at {encoder_path}!")
            exit(1)

        print(f"Loading GCN model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            print("⚠️ Trying with custom_objects...")
            try:
                from src.debug.train_gcn import GraphConv, STGCN_Block
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'GraphConv': GraphConv, 'STGCN_Block': STGCN_Block}
                )
            except ImportError:
                print("❌ Could not import from train_gcn.py")
                exit(1)

        # Warm-up: chạy 1 lần giả để compile graph, tránh lag lần predict đầu
        dummy = np.zeros((1, 30, 75, 3), dtype=np.float32)
        _ = self.model(dummy, training=False)
        print("✓ Model warmed up.")

        self.labels = np.load(encoder_path, allow_pickle=True)
        print(f"✓ Labels: {self.labels}")

    def _init_detectors(self):
        models_url = {
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            # Face detector đã bỏ — GCN không dùng face, chạy nó mỗi frame lãng phí ~30% thời gian
        }
        import urllib.request
        for name, url in models_url.items():
            if not os.path.exists(name):
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, name)
                except Exception as e:
                    print(f"⚠️ Download failed: {e}")

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5
            )
        )

        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_options)
        )

    def extract_keypoints(self, hand_result, pose_result):
        """
        Trích xuất keypoints: Pose(99) + Face zeros(1434) + Hands(126) = 1659.
        Face = zeros vì không chạy face detector.
        Index trong vector vẫn giữ nguyên để nhất quán với collect & train.
        """
        keypoints = []

        # 1. Pose (99)
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        # 2. Face — zeros placeholder (1434)
        # Không chạy face detector để tiết kiệm thời gian, GCN không dùng face
        keypoints.extend([0.0] * 1434)

        # 3. Hands — LEFT trước, RIGHT sau (nhất quán với collect)
        left_hand  = [0.0] * 63
        right_hand = [0.0] * 63

        if hand_result.hand_landmarks and hand_result.handedness:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                hand_kps = []
                for lm in hand_landmarks:
                    hand_kps.extend([lm.x, lm.y, lm.z])
                label = hand_result.handedness[i][0].category_name
                if label == "Left":
                    left_hand = hand_kps
                else:
                    right_hand = hand_kps

        keypoints.extend(left_hand)
        keypoints.extend(right_hand)

        return np.array(keypoints, dtype=np.float32)

    def preprocess_for_gcn(self, buffer_data):
        """(30, 1659) normalized → (1, 30, 75, 3)"""
        seq      = np.array(buffer_data, dtype=np.float32)
        pose     = seq[:, 0:99]
        hands    = seq[:, 1533:1659]
        skeleton = np.concatenate([pose, hands], axis=1)  # (30, 225)
        return skeleton.reshape(1, 30, 75, 3)

    def draw_debug(self, frame, pose_result, hand_result):
        h, w, _ = frame.shape
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 200, 200), -1)
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 80), -1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return

        print("\n=== VSL GCN REAL-TIME TEST ===")
        print(f"Predict every {self.PREDICT_EVERY} frames | Conf ≥ {self.CONF_THRESHOLD}")
        print("Q: Quit | R: Reset buffer")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect — chỉ Hand + Pose (bỏ Face để tăng tốc)
            hand_res = self.hand_detector.detect(mp_image)
            pose_res = self.pose_detector.detect(mp_image)

            frame = self.draw_debug(frame, pose_res, hand_res)

            # Extract → Normalize → Buffer
            kps      = self.extract_keypoints(hand_res, pose_res)
            norm_kps = self.normalize_keypoints(kps)
            self.buffer.append(norm_kps)
            self.frame_count += 1

            # ==========================================
            # PREDICT LOGIC
            # ==========================================
            if len(self.buffer) == 30:
                seq_array = np.array(self.buffer, dtype=np.float32)

                # Tính variance trên phần tay (index 1533:1659) trong buffer đã normalize
                hand_data       = seq_array[:, 1533:1659]
                motion_variance = float(np.var(hand_data[hand_data != 0])) if np.any(hand_data != 0) else 0.0

                currently_moving = motion_variance > self.MOTION_THRESHOLD
                if currently_moving:
                    self.in_motion = True

                if self.in_motion and (self.frame_count % self.PREDICT_EVERY == 0):
                    gcn_input = self.preprocess_for_gcn(self.buffer)

                    # model() nhanh hơn model.predict() ~3-5x với batch size = 1
                    pred = self.model(gcn_input, training=False).numpy()[0]

                    conf = float(np.max(pred))
                    idx  = int(np.argmax(pred))

                    # Debug top 3 in terminal
                    top3      = np.argsort(pred)[-3:][::-1]
                    debug_str = " | ".join([f"{self.labels[i]}:{pred[i]:.2f}" for i in top3])
                    print(f"\r[Top3] {debug_str}   ", end="", flush=True)

                    if conf >= self.CONF_THRESHOLD:
                        self.last_sign = self.labels[idx]
                        self.last_conf = conf

                if not currently_moving and self.in_motion:
                    self.in_motion = False

            # ==========================================
            # UI
            # ==========================================
            cv2.rectangle(frame, (0, 0), (500, 115), (15, 15, 15), -1)

            if self.last_sign:
                cv2.putText(frame, self.last_sign.upper(), (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 80), 2)
                cv2.putText(frame, f"Conf: {self.last_conf:.1%}", (20, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            else:
                cv2.putText(frame, "Waiting for sign...", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (130, 130, 130), 2)

            # Status bar
            buf_color    = (0, 255, 100) if len(self.buffer) == 30 else (80, 80, 255)
            motion_color = (0, 200, 255) if self.in_motion else (60, 60, 60)
            cv2.putText(frame, f"Buffer:{len(self.buffer)}/30", (10, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, buf_color, 1)
            cv2.putText(frame, f"Motion:{'ON' if self.in_motion else 'OFF'}", (150, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
            cv2.putText(frame, f"Frame:{self.frame_count}", (260, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow('VSL GCN Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.buffer.clear()
                self.last_sign = ""
                self.last_conf = 0.0
                self.in_motion = False
                print("\n🔄 Reset!")

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Done.")


if __name__ == '__main__':
    tester = VSLGCNTester()
    tester.run()
