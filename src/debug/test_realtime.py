"""
VSL Real-time Tester - Holistic (Hands + Face + Pose)
MediaPipe 0.10.32+ Task API
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VSLTester:
    def __init__(self):
        # Find model path
        model_paths = glob_models()
        if not model_paths:
            print("Model not found! Please train the model first.")
            exit(1)
            
        model_path = model_paths[0] # Pick the first found model (or best)
        # Try to find label encoder
        encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder.npy')
        
        if not os.path.exists(encoder_path):
             print(f"Label encoder not found at {encoder_path}")
             exit(1)

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(encoder_path, allow_pickle=True)
        
        # Initialize Detectors
        print("Initializing MediaPipe Holistic (Task API)...")
        self._init_detectors()
        
        # Buffer
        self.buffer = deque(maxlen=30)
        
        print("✓ Model loaded")
        print(f"✓ Signs: {self.labels}")

    def _init_detectors(self):
        # Download models if needed (same logic as collector)
        models = {
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        }
        import urllib.request
        for name, url in models.items():
            if not os.path.exists(name):
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, name)
                except Exception as e:
                    print(f"❌ Failed to download {name}: {e}")
                    exit(1)

        # Hand
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # Face
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

        # Pose
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
    
    def extract_keypoints(self, hand_result, face_result, pose_result):
        """Extract combined keypoints: Pose(99) + Face(1434) + Hands(126) = 1659"""
        keypoints = []
        
        # 1. Pose
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 99)
            
        # 2. Face
        if face_result.face_landmarks:
            for landmark in face_result.face_landmarks[0]:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 1434)
            
        # 3. Hands
        hand_kps = []
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    hand_kps.extend([landmark.x, landmark.y, landmark.z])
        while len(hand_kps) < 126:
            hand_kps.extend([0] * 63)
        keypoints.extend(hand_kps[:126])
        
        return keypoints
    
    
    def draw_landmarks(self, frame, hand_result, face_result, pose_result):
        h, w, _ = frame.shape
        
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 2, (255, 0, 0), -1)

        if face_result.face_landmarks:
            # Draw sparse face points
            for i in range(0, len(face_result.face_landmarks[0]), 20):
                landmark = face_result.face_landmarks[0][i]
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 255), -1)

        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 255, 0), -1)
                    
        return frame
    
    def predict(self):
        """Predict sign with shape adaptation"""
        if len(self.buffer) < 30:
            return None, 0.0
        
        seq = np.array(list(self.buffer))
        
        # Lấy shape input mong đợi của model
        expected_shape = self.model.input_shape[1:] # (30, features)
        expected_features = expected_shape[1]
        current_features = seq.shape[1]

        # Xử lý lệch shape (Holistic vs Hand-only)
        if current_features != expected_features:
            if current_features == 1659 and expected_features == 126:
                # Nếu đang chạy Holistic (1659) nhưng model cũ (126)
                # Hand keypoints nằm ở cuối vector Holistic
                seq = seq[:, -126:] 
            else:
                print(f"\r⚠️ Shape mismatch: Input {seq.shape} vs Model {expected_shape}", end="")
                return None, 0.0

        seq = np.expand_dims(seq, axis=0)
        
        try:
            pred = self.model.predict(seq, verbose=0)[0]
            conf = np.max(pred)
            idx = np.argmax(pred)
            
            if conf < 0.6:
                return None, conf
            
            return self.labels[idx], conf
        except Exception as e:
            print(f"Error predicting: {e}")
            return None, 0.0
    
    def run(self):
        # Try camera 0 first, then 1
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("\n=== VSL HOLISTIC REAL-TIME TEST ===")
        print("Press 'Q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect
            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_detector.detect(mp_image)
            pose_result = self.pose_detector.detect(mp_image)
            
            # Draw
            frame = self.draw_landmarks(frame, hand_result, face_result, pose_result)
            
            # Extract & Predict
            kps = self.extract_keypoints(hand_result, face_result, pose_result)
            self.buffer.append(kps)
            
            sign, conf = self.predict()
            
            # UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)
            
            if sign:
                cv2.putText(frame, sign.upper(), (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"Conf: {conf:.0%}", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Waiting...", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            cv2.putText(frame, f"Buffer: {len(self.buffer)}/30", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('VSL Holistic Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

def glob_models():
    import glob
    return glob.glob('../models/*.h5') + glob.glob('models/*.h5')

def main():
    tester = VSLTester()
    tester.run()
if __name__ == '__main__':
    main()
