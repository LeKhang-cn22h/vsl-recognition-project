"""
VSL Real-time Tester - GCN Version v3

FIX LỖI LOAD MODEL (Lambda layer):
- Model cũ được save với Lambda layer → Keras chặn vì lý do bảo mật
- Giải pháp tạm: load với safe_mode=False (an toàn vì model do chính mình train)
- Giải pháp vĩnh viễn: train lại với train_gcn.py mới (đã dùng NodePool thay Lambda)
                        Model mới sẽ load bình thường không cần safe_mode=False

FIX KHÁC (giữ nguyên):
- Hierarchical normalize: Pose theo vai, Hands theo wrist
- motion_variance theo trục thời gian (axis=0)
- Bỏ Face Detector, model() thay predict(), warm-up
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Index mapping trong vector reshape (553, 3)
POSE_START  = 0
POSE_END    = 33
LHAND_START = 511   # 33 + 478
LHAND_END   = 532   # 511 + 21
RHAND_START = 532
RHAND_END   = 553   # 532 + 21


# ============================================================
# CUSTOM LAYERS — định nghĩa tại đây, độc lập với train_gcn.py
# ============================================================
class GraphConv(layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels})
        return config

    def build(self, input_shape):
        self.nodes       = input_shape[2]
        self.in_channels = input_shape[3]
        self.A = self.add_weight(
            name="adjacency_matrix",
            shape=(self.nodes, self.nodes),
            initializer="identity",
            regularizer=keras.regularizers.l2(0.001),
            trainable=True
        )
        self.W = self.add_weight(
            name="weight_matrix",
            shape=(self.in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        A_norm = tf.nn.softmax(self.A, axis=-1)
        x = tf.einsum('vw,btwc->btvc', A_norm, inputs)
        x = tf.matmul(x, self.W)
        return tf.nn.relu(x)


class STGCN_Block(layers.Layer):
    def __init__(self, out_channels, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.dropout_rate = dropout
        self.gcn        = GraphConv(out_channels)
        self.tcn        = layers.Conv2D(out_channels, kernel_size=(9, 1),
                                        padding='same', activation='relu')
        self.dropout    = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
        self.residual   = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')
        self.add        = layers.Add()

    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels, 'dropout': self.dropout_rate})
        return config

    def call(self, inputs, training=None):
        x   = self.gcn(inputs)
        x   = self.tcn(x)
        x   = self.batch_norm(x, training=training)
        x   = self.dropout(x, training=training)
        res = self.residual(inputs)
        return self.add([x, res])


class NodePool(layers.Layer):
    """
    Pool theo chiều Node (axis=2), GIỮ chiều Time.
    (B, T, V, C) → (B, T, C)
    Thay thế Lambda để load/save an toàn (model mới từ train_gcn.py).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)


# ============================================================
# TESTER
# ============================================================
class VSLGCNTester:
    PREDICT_EVERY    = 20
    MOTION_THRESHOLD = 0.0005
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

    # ============================================================
    # NORMALIZE — Hierarchical, giống hệt auto_collect_data.py
    # ============================================================
    def normalize_keypoints(self, keypoints):
        """
        Pose[0:33]     → tâm = midpoint vai (kps[11], kps[12]), scale = dist vai
        LHand[511:532] → tâm = wrist (kps[511]), scale = wrist→middle_mcp
        RHand[532:553] → tâm = wrist (kps[532]), scale = wrist→middle_mcp
        """
        kps = np.array(keypoints, dtype=np.float32).reshape(-1, 3)

        # 1. POSE
        left_shoulder   = kps[11].copy()
        right_shoulder  = kps[12].copy()
        shoulder_ok     = np.any(left_shoulder != 0) and np.any(right_shoulder != 0)
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_dist   = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_ok and shoulder_dist > 1e-6:
            detected = np.any(kps[POSE_START:POSE_END] != 0, axis=1)
            idxs     = np.where(detected)[0] + POSE_START
            kps[idxs] = (kps[idxs] - shoulder_center) / shoulder_dist

        fallback_scale = shoulder_dist if (shoulder_ok and shoulder_dist > 1e-6) else 0.3

        # 2. LEFT HAND
        left_wrist = kps[LHAND_START].copy()
        if np.any(left_wrist != 0):
            lmcp       = kps[LHAND_START + 9].copy()
            hand_scale = np.linalg.norm(left_wrist - lmcp)
            hand_scale = hand_scale if hand_scale > 1e-6 else fallback_scale * 0.3
            detected   = np.any(kps[LHAND_START:LHAND_END] != 0, axis=1)
            idxs       = np.where(detected)[0] + LHAND_START
            kps[idxs]  = (kps[idxs] - left_wrist) / hand_scale

        # 3. RIGHT HAND
        right_wrist = kps[RHAND_START].copy()
        if np.any(right_wrist != 0):
            rmcp       = kps[RHAND_START + 9].copy()
            hand_scale = np.linalg.norm(right_wrist - rmcp)
            hand_scale = hand_scale if hand_scale > 1e-6 else fallback_scale * 0.3
            detected   = np.any(kps[RHAND_START:RHAND_END] != 0, axis=1)
            idxs       = np.where(detected)[0] + RHAND_START
            kps[idxs]  = (kps[idxs] - right_wrist) / hand_scale

        return kps.flatten()

    def _load_resources(self):
        model_path = '../models/best_gcn_model.h5'
        if not os.path.exists(model_path):
            model_path = 'models/best_gcn_model.h5'
        if not os.path.exists(model_path):
            print("❌ Model not found! Run train_gcn.py first.")
            exit(1)

        encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder_gcn.npy')
        if not os.path.exists(encoder_path):
            print(f"❌ Label encoder not found!")
            exit(1)

        print(f"Loading model: {model_path}")

        # Custom objects bao gồm NodePool cho model mới,
        # GraphConv + STGCN_Block cho cả model cũ lẫn mới
        custom_objects = {
            'GraphConv':   GraphConv,
            'STGCN_Block': STGCN_Block,
            'NodePool':    NodePool,
        }

        try:
            # Thử load bình thường trước (model mới train bằng train_gcn.py đã fix)
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
            print("✓ Model loaded.")
        except Exception as e:
            if 'Lambda' in str(e) or 'safe_mode' in str(e):
                # Model cũ còn Lambda layer → cần safe_mode=False
                print("⚠️  Model cũ có Lambda layer. Loading với safe_mode=False...")
                print("    → Hãy train lại với train_gcn.py mới để fix vĩnh viễn.")
                try:
                    self.model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=custom_objects,
                        safe_mode=False          # Cho phép Lambda, an toàn vì model do mình train
                    )
                    print("✓ Model loaded (safe_mode=False).")
                except Exception as e2:
                    print(f"❌ Load thất bại hoàn toàn: {e2}")
                    exit(1)
            else:
                print(f"❌ Load thất bại: {e}")
                exit(1)

        # Warm-up: compile TF graph
        dummy = np.zeros((1, 30, 75, 3), dtype=np.float32)
        _ = self.model(dummy, training=False)
        print("✓ Model warmed up.")

        self.labels = np.load(encoder_path, allow_pickle=True)
        print(f"✓ Labels: {self.labels}")

    def _init_detectors(self):
        models_url = {
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        }
        import urllib.request
        for name, url in models_url.items():
            if not os.path.exists(name):
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, name)
                except Exception as e:
                    print(f"⚠️ {e}")

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
        """Pose(99) + Face_zeros(1434) + LHand(63) + RHand(63) = 1659"""
        keypoints = []

        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        keypoints.extend([0.0] * 1434)  # Face zeros

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
        """(30, 1659) → (1, 30, 75, 3)"""
        seq      = np.array(buffer_data, dtype=np.float32)
        pose     = seq[:, 0:99]
        hands    = seq[:, 1533:1659]
        skeleton = np.concatenate([pose, hands], axis=1)
        return skeleton.reshape(1, 30, 75, 3)

    def compute_motion_variance(self, seq_array):
        """Variance theo trục thời gian (axis=0) — đo CHUYỂN ĐỘNG, không phải giá trị."""
        hand_data = seq_array[:, 1533:1659]
        return float(np.var(hand_data, axis=0).mean())

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

        print("\n=== VSL GCN REAL-TIME TEST v3 ===")
        print(f"Predict every {self.PREDICT_EVERY} frames | Conf ≥ {self.CONF_THRESHOLD}")
        print("Q: Quit | R: Reset")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_res = self.hand_detector.detect(mp_image)
            pose_res = self.pose_detector.detect(mp_image)

            frame = self.draw_debug(frame, pose_res, hand_res)

            kps      = self.extract_keypoints(hand_res, pose_res)
            norm_kps = self.normalize_keypoints(kps)
            self.buffer.append(norm_kps)
            self.frame_count += 1

            if len(self.buffer) == 30:
                seq_array       = np.array(self.buffer, dtype=np.float32)
                motion_variance = self.compute_motion_variance(seq_array)

                currently_moving = motion_variance > self.MOTION_THRESHOLD
                if currently_moving:
                    self.in_motion = True

                if self.in_motion and (self.frame_count % self.PREDICT_EVERY == 0):
                    gcn_input = self.preprocess_for_gcn(self.buffer)
                    pred      = self.model(gcn_input, training=False).numpy()[0]

                    conf = float(np.max(pred))
                    idx  = int(np.argmax(pred))

                    top3      = np.argsort(pred)[-3:][::-1]
                    debug_str = " | ".join([f"{self.labels[i]}:{pred[i]:.2f}" for i in top3])
                    print(f"\r[Top3] {debug_str} [Var:{motion_variance:.5f}]   ",
                          end="", flush=True)

                    if conf >= self.CONF_THRESHOLD:
                        self.last_sign = self.labels[idx]
                        self.last_conf = conf

                if not currently_moving and self.in_motion:
                    self.in_motion = False

            cv2.rectangle(frame, (0, 0), (520, 120), (15, 15, 15), -1)

            if self.last_sign:
                cv2.putText(frame, self.last_sign.upper(), (20, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 80), 2)
                cv2.putText(frame, f"Conf: {self.last_conf:.1%}", (20, 102),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            else:
                cv2.putText(frame, "Waiting for sign...", (20, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (130, 130, 130), 2)

            buf_color    = (0, 255, 100) if len(self.buffer) == 30 else (80, 80, 255)
            motion_color = (0, 200, 255) if self.in_motion else (60, 60, 60)

            cv2.putText(frame, f"Buffer:{len(self.buffer)}/30", (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, buf_color, 1)
            cv2.putText(frame, f"Motion:{'ON' if self.in_motion else 'OFF'}", (160, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
            cv2.putText(frame, f"Frame:{self.frame_count}", (290, 445),
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
