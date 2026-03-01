"""
VSL Data Collector - Holistic (Hands + Face + Pose)
MediaPipe 0.10.32+ Task API
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

class VSLDataCollector:
    def __init__(self, output_dir=RAW_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing MediaPipe Holistic (Task API)...")
        
        # 1. Setup Models
        self._setup_models()
        
        # 2. Initialize Detectors
        self._init_hand_detector()
        self._init_face_detector()
        self._init_pose_detector()
        
        # Metadata
        self.metadata = {
            'signs': {},
            'total_samples': 0,
            'feature_schema': ['pose', 'face', 'left_hand', 'right_hand']
        }
        
        print("✓ Holistic Detectors Initialized")
        
    def _setup_models(self):
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
                    print(f"✓ {name} downloaded")
                except Exception as e:
                    print(f"❌ Failed to download {name}: {e}")

    def _init_hand_detector(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            # độ nhập diện bàn tay thấp hơn .5 thì bỏ
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # tạo object nhận diện tay
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Metadata
        self.metadata = {
            'signs': {},
            'total_samples': 0
        }
        
        print("✓ MediaPipe 0.10.32 initialized with Task API")
    # biến output mediapipe thành vector số
    def extract_keypoints(self, detection_result):
        """Extract 126 features from detection result"""
        keypoints = []
        # Lấy toàn bộ tọa độ các điểm
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 99)
            
        # 2. Face (478 landmarks)
        if face_result.face_landmarks:
            for landmark in face_result.face_landmarks[0]:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 1434)
            
        # 3. Hands (21 * 2 landmarks)
        # Note: We need to distinguish left/right if possible, but for simplicity
        # we'll stick to the previous logic of just appending what we find, padding if missing.
        # Ideally, we should check handedness, but that requires more complex logic.
        hand_kps = []
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    hand_kps.extend([landmark.x, landmark.y, landmark.z])
        
        # Pad hands to 126
        while len(hand_kps) < 126:
            hand_kps.extend([0] * 63)
            
        keypoints.extend(hand_kps[:126])
        
        return keypoints
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints.
        Structure: [Pose(99) | Face(1434) | Hands(126)]
        """
        # TODO: Implement sophisticated normalization (e.g., centering face/pose)
        # For now, we return raw/relative coordinates as extracted (already normalized 0-1 by MediaPipe)
        # Exception: Z-coordinates in MediaPipe are not 0-1, they are relative depth.
        return keypoints # Returning flattened list
    
    def draw_landmarks(self, frame, hand_result, face_result, pose_result):
        h, w, _ = frame.shape
        
        # Draw Pose
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 2, (255, 0, 0), -1)

        # Draw Face (Contours only to avoid clutter)
        if face_result.face_landmarks:
            # Drawing 478 points is too messy. Let's draw indices for eyes/lips if needed.
            # For now, just a bounding box or simple dots
            for i in range(0, len(face_result.face_landmarks[0]), 10): # Draw every 10th point
                landmark = face_result.face_landmarks[0][i]
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 255), -1)

        # Draw Hands
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 255, 0), -1)
                    
        return frame

    def collect_sign(self, sign_name, num_samples=30, sequence_length=30):
        cap = cv2.VideoCapture(1) # Try 0 if 1 doesn't work
        if not cap.isOpened():
            print("Cannot open camera!")
            return 0
            
        sign_dir = os.path.join(self.output_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        sample_count = 0
        recording = False
        sequence = []
        
        print(f"\nCollecting: {sign_name.upper()} | Target: {num_samples}")
        print("Press 'S' to Start | 'Q' to Quit")
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect ALL
            # Note: Running sequentially might reduce FPS
            hand_result = self.hand_detector.detect(mp_image)
            face_result = self.face_detector.detect(mp_image)
            pose_result = self.pose_detector.detect(mp_image)
            
            # Draw
            frame = self.draw_landmarks(frame, hand_result, face_result, pose_result)
            
            # UI Info
            cv2.putText(frame, f"Sign: {sign_name} ({sample_count}/{num_samples})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if recording:
                cv2.putText(frame, "RECORDING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"{len(sequence)}/{sequence_length}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Press 'S' to Start", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Collection Logic
            if recording:
                keypoints = self.extract_keypoints(hand_result, face_result, pose_result)
                sequence.append(keypoints)
                
                if len(sequence) >= sequence_length:
                    save_path = os.path.join(sign_dir, f"{sign_name}_{int(time.time())}.npy")
                    np.save(save_path, np.array(sequence))
                    sample_count += 1
                    sequence = []
                    recording = False
                    print(f"✓ Saved sample {sample_count}")
                    time.sleep(0.5)
            
            cv2.imshow('VSL Holistic Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s') and not recording:
                recording = True
                sequence = []
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Update metadata
        if sign_name not in self.metadata['signs']:
            self.metadata['signs'][sign_name] = {'count': 0}
        self.metadata['signs'][sign_name]['count'] += sample_count
        self.metadata['total_samples'] += sample_count
        
        return sample_count

    def save_metadata(self):
        path = os.path.join(self.output_dir, 'metadata.json')
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

def main():
    collector = VSLDataCollector()
    while True:
        sign = input("\nEnter sign name (or 'q' to quit): ").strip()
        if sign.lower() == 'q' or sign == '': break
        
        try:
            samples = int(input("Number of samples (default 30): ") or "30")
        except ValueError:
            samples = 30
            
        collector.collect_sign(sign, samples)
        collector.save_metadata()

if __name__ == '__main__':
    main()
