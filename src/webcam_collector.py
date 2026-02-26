"""
webcam_collector.py - Thu thập video ngôn ngữ ký hiệu VSL
==========================================================
Cách chạy:
    python webcam_collector.py

Menu:
    1. Xem thống kê video
    2. Tạo nhãn mới và thu video
    3. Tiếp tục thu video cho nhãn có sẵn
    4. Thu video IDLE (nghỉ / không ký hiệu)
    5. Lưu và thoát
"""

import cv2
import json
import os
import time
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from collector import (
    init_hf, upload_to_hf,
    FullBodyDrawer, draw_text_bg, lm_to_px,
    FramingChecker,
    FacialExpressionAnalyzer,
    InteractionVisualizer,
)

# ── Download MediaPipe models ──────────────────────────────

MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
}

def download_model(filename):
    if os.path.exists(filename):
        return filename
    print(f"  Dang tai {filename} ...")
    urllib.request.urlretrieve(MODEL_URLS[filename], filename)
    return filename


# ══════════════════════════════════════════════════════════
# LỚP CHÍNH
# ══════════════════════════════════════════════════════════

class WebcamVideoCollector:

    # Hằng số điều khiển auto mode
    COUNTDOWN_SECS         = 5
    RELAXED_FRAMES_TO_STOP = 15   # ~0.5s @ 30fps
    COOLDOWN_SECS          = 2.0

    def __init__(self, output_dir='data/videos'):
        self.output_dir    = output_dir
        self.metadata_path = os.path.join(output_dir, 'metadata.json')
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = self._load_meta()

        init_hf()   # Khởi tạo HuggingFace upload

        print("\n" + "="*60)
        print(" KHOI TAO MEDIAPIPE DETECTORS ".center(60))
        print("="*60)

        hand_m = download_model('hand_landmarker.task')
        pose_m = download_model('pose_landmarker_heavy.task')
        face_m = download_model('face_landmarker.task')

        # Kết quả callback lưu vào đây, main loop đọc ra
        self._latest = dict(pose=None, face=None, hands=None, blendshapes=None)
        self._ts = 0

        print("  Khoi tao PoseLandmarker ...")
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_pose))

        print("  Khoi tao HandLandmarker ...")
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_hand))

        print("  Khoi tao FaceLandmarker (+ Blendshapes) ...")
        self.face_detector = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                result_callback=self._on_face))

        print("  Tat ca detector da san sang!\n")

    # ── MediaPipe callbacks ───────────────────────────────

    def _on_pose(self, result, image, ts):
        self._latest['pose'] = (result.pose_landmarks[0]
                                if result.pose_landmarks else None)

    def _on_hand(self, result, image, ts):
        left = right = None
        if result.hand_landmarks and result.handedness:
            for i, hlms in enumerate(result.hand_landmarks):
                cat = result.handedness[i][0].category_name
                if cat == 'Left': right = hlms
                else:             left  = hlms
        self._latest['hands'] = (left, right)

    def _on_face(self, result, image, ts):
        self._latest['face'] = (result.face_landmarks[0]
                                if result.face_landmarks else None)
        self._latest['blendshapes'] = (result.face_blendshapes[0]
                                       if result.face_blendshapes else None)

    # ── Metadata ─────────────────────────────────────────

    def _load_meta(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return dict(labels={}, total_videos=0,
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def _save_meta(self):
        self.metadata['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    # ── Display names ─────────────────────────────────────

    def _dn_path(self):
        return os.path.normpath(
            os.path.join(self.output_dir, '..', 'processed', 'display_names.json'))

    def _save_display_name(self, label_key, viet_name):
        path = self._dn_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dn = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                dn = json.load(f)
        if label_key not in dn:
            dn[label_key] = viet_name
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dn, f, indent=2, ensure_ascii=False)
            print(f"  Da luu: '{label_key}' → '{viet_name}'")

    # ── Statistics ────────────────────────────────────────

    def show_statistics(self):
        print("\n" + "="*60)
        print(" THONG KE VIDEO DA THU ".center(60))
        print("="*60)
        if not self.metadata['labels']:
            print("\n  Chua co video nao")
        else:
            total = 0
            print(f"\n  {'Nhan':<30} {'So video':<15} {'Duong dan'}")
            print("  " + "-"*65)
            for lb, info in sorted(self.metadata['labels'].items()):
                n = info.get('num_videos', 0)
                print(f"  {lb:<30} {n:<15} {info.get('path','')}")
                total += n
            print("  " + "-"*65)
            print(f"  {'TONG CONG':<30} {total}")
        print(f"\n  Cap nhat: {self.metadata.get('updated_at','N/A')}")
        print("="*60)

    # ── Helpers: kiểm tra tay thả lỏng ───────────────────

    @staticmethod
    def _hands_relaxed(pose_lms, left_hand_lms, right_hand_lms) -> bool:
        """True nếu không thấy tay, hoặc cổ tay nằm dưới / ngang hông."""
        if left_hand_lms is None and right_hand_lms is None:
            return True
        if pose_lms is None:
            return False

        # Tính hip_y
        hip_y = None
        if pose_lms[23].visibility > 0.4 and pose_lms[24].visibility > 0.4:
            hip_y = (pose_lms[23].y + pose_lms[24].y) / 2
        elif pose_lms[23].visibility > 0.4:
            hip_y = pose_lms[23].y
        elif pose_lms[24].visibility > 0.4:
            hip_y = pose_lms[24].y
        elif pose_lms[11].visibility > 0.4 and pose_lms[12].visibility > 0.4:
            hip_y = (pose_lms[11].y + pose_lms[12].y) / 2 + 0.25
        else:
            return False

        margin = 0.03
        left_ok  = True
        right_ok = True
        if pose_lms[15].visibility > 0.4:
            left_ok  = pose_lms[15].y > (hip_y - margin)
        if pose_lms[16].visibility > 0.4:
            right_ok = pose_lms[16].y > (hip_y - margin)
        return left_ok and right_ok

    # ── UI helpers ────────────────────────────────────────

    def _draw_warnings(self, frame, fr, w, h):
        if fr['ok']:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,255,0), 3)
            draw_text_bg(frame, "GOC QUAY: OK", (10, h-60),
                         scale=0.6, color=(0,255,0), bg=(0,50,0))
        else:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 4)
            y = h - 60 - (len(fr['warnings'])-1)*30
            for w_txt in fr['warnings']:
                draw_text_bg(frame, f"! {w_txt}", (10, y),
                             scale=0.55, color=(0,0,255), bg=(50,0,0))
                y += 30

        det = fr['details']
        items = [('Mat', det['face_visible']),
                 ('Than', det['upper_body_visible']),
                 ('Tay T', det['left_arm_visible']),
                 ('Tay P', det['right_arm_visible']),
                 ('Ban tay T', det['left_hand_visible']),
                 ('Ban tay P', det['right_hand_visible'])]
        x0 = w - 130
        for i, (nm, ok) in enumerate(items):
            c = (0,255,0) if ok else (0,0,255)
            cv2.putText(frame, f"{'[OK]' if ok else '[X] '} {nm}",
                        (x0, 70+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    def _draw_expression(self, frame, expr, w, h):
        if expr is None:
            return
        px, py = 10, 70
        src = expr.get('source', '?')
        draw_text_bg(frame,
                     f"Bieu cam: {expr['expression_label']} [{src}]",
                     (px, py), scale=0.55, color=(255,255,0), bg=(40,40,40))
        lines = [
            f"Mieng: {'Mo' if expr['mouth_open']>0.3 else 'Dong'} "
            f"({expr['mouth_open']:.2f})  Cuoi:{expr['mouth_smile']:.2f}",
            f"Mat T:{expr['left_eye_open']:.2f}  Mat P:{expr['right_eye_open']:.2f}"
            f"  Wide:{expr.get('eye_wide',0):.2f}  Squint:{expr.get('eye_squint',0):.2f}",
            f"May len:{expr.get('brow_up',0):.2f}  May xuong:{expr.get('brow_down',0):.2f}",
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (px, py+22+i*17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200,200,200), 1)

    def _draw_interactions(self, frame, interactions, w, h):
        if not interactions:
            return
        y = 195
        draw_text_bg(frame, "TUONG TAC:", (10, y),
                     scale=0.55, color=(0,255,255), bg=(40,40,40))
        for i, txt in enumerate(interactions):
            cv2.putText(frame, f">> {txt}", (10, y+22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

    # ── STATE DISPLAY helpers ─────────────────────────────

    def _draw_countdown(self, frame, w, h, elapsed_cd, num_color):
        remaining  = max(0, self.COUNTDOWN_SECS - elapsed_cd)
        count_text = str(int(remaining) + 1)
        fs = 4.0
        (ctw, cth), _ = cv2.getTextSize(
            count_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 6)
        cx = (w - ctw) // 2
        cy = (h + cth) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (cx-40, cy-cth-30),
                      (cx+ctw+40, cy+30), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, count_text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, num_color, 6)
        cv2.putText(frame, "CHUAN BI...", (w//2-80, cy+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        bar_y = cy + 70
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+bar_w, bar_y+12), (80,80,80), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+int(bar_w*(elapsed_cd/self.COUNTDOWN_SECS)), bar_y+12),
                      num_color, -1)

    def _draw_recording(self, frame, w, h, elapsed, frame_count,
                         relaxed_cnt):
        cv2.putText(frame, f"REC {elapsed:.1f}s | {frame_count}f",
                    (w//2-80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if int(elapsed*2) % 2 == 0:
            cv2.circle(frame, (w//2-100, 20), 8, (0,0,255), -1)
        if relaxed_cnt > 3:
            ratio = relaxed_cnt / self.RELAXED_FRAMES_TO_STOP
            cv2.putText(frame,
                        f"Tha tay... dung sau {self.RELAXED_FRAMES_TO_STOP - relaxed_cnt} frames",
                        (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            bar_w = 300
            bar_x = (w - bar_w) // 2
            cv2.rectangle(frame, (bar_x, h//2+15),
                          (bar_x+bar_w, h//2+25), (80,80,80), -1)
            cv2.rectangle(frame, (bar_x, h//2+15),
                          (bar_x+int(bar_w*ratio), h//2+25), (0,200,255), -1)

    # ══════════════════════════════════════════════════════
    # THU THẬP VIDEO
    # ══════════════════════════════════════════════════════

    def collect_label(self, label_name: str):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Khong the mo webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        label_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)
        video_count = len([f for f in os.listdir(label_dir)
                           if f.endswith('.mp4')])

        # ── State machine: idle → countdown → recording → idle ──
        state          = 'idle'
        video_writer   = None
        frame_count    = 0
        start_time     = 0
        countdown_start = 0
        relaxed_cnt    = 0
        last_stop_time = 0
        fp             = None
        show_mesh      = True
        auto_mode      = True

        print(f"\n  Nhan: {label_name.upper()} | "
              f"Da co: {video_count} video | {width}x{height}@{fps}")
        print("  [SPACE] Thu cong  [A] Auto  [M] Mesh  [Q] Thoat\n")

        self._ts = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame       = cv2.flip(frame, 1)
            clean_frame = frame.copy()   # frame GỐC để ghi video (không overlay)
            h, w        = frame.shape[:2]
            now         = time.time()

            # ── Gửi đến MediaPipe ──
            self._ts += 33
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for det in [self.pose_detector, self.hand_detector, self.face_detector]:
                try: det.detect_async(mp_img, self._ts)
                except Exception: pass

            # ── Đọc kết quả ──
            pose_lms  = self._latest['pose']
            face_lms  = self._latest['face']
            blends    = self._latest['blendshapes']
            left_h, right_h = self._latest['hands'] or (None, None)

            # ── Vẽ keypoints ──
            FullBodyDrawer.draw_pose(frame, pose_lms, w, h)
            if show_mesh:
                FullBodyDrawer.draw_face_mesh(frame, face_lms, w, h)
            FullBodyDrawer.draw_hand(frame, left_h,  w, h, 'L')
            FullBodyDrawer.draw_hand(frame, right_h, w, h, 'R')

            # ── Phân tích ──
            framing = FramingChecker.check(
                pose_lms, face_lms, (left_h, right_h), w, h)
            self._draw_warnings(frame, framing, w, h)

            expr = (FacialExpressionAnalyzer.analyze_blendshapes(blends)
                    if blends else
                    FacialExpressionAnalyzer.analyze_landmarks(face_lms, w, h))
            self._draw_expression(frame, expr, w, h)

            frame, interactions = InteractionVisualizer.draw(
                frame, pose_lms, face_lms, left_h, right_h, w, h)
            self._draw_interactions(frame, interactions, w, h)

            relaxed = self._hands_relaxed(pose_lms, left_h, right_h)

            # ══════════════════════════════════════════════
            # STATE MACHINE
            # ══════════════════════════════════════════════
            if auto_mode:
                if state == 'idle':
                    in_cooldown = (now - last_stop_time) < self.COOLDOWN_SECS
                    if framing['ok'] and not relaxed and not in_cooldown:
                        state = 'countdown'
                        countdown_start = now
                        relaxed_cnt = 0
                        print(f"  Auto: OK → dem nguoc {self.COUNTDOWN_SECS}s...")

                elif state == 'countdown':
                    elapsed_cd = now - countdown_start
                    if elapsed_cd >= self.COUNTDOWN_SECS:
                        state, fp, video_writer, frame_count, start_time = \
                            self._start_recording(label_name, label_dir,
                                                  video_count, fps, width, height, now)
                        print(f"  Auto: BAT DAU video {video_count+1}")
                    elif not framing['ok'] or relaxed:
                        state = 'idle'
                        print("  Auto: Huy dem nguoc")

                elif state == 'recording':
                    relaxed_cnt = relaxed_cnt + 1 if relaxed else 0
                    if relaxed_cnt >= self.RELAXED_FRAMES_TO_STOP:
                        video_count, last_stop_time = \
                            self._stop_recording(video_writer, video_count,
                                                 frame_count, now - start_time,
                                                 fp, label_name, now)
                        video_writer = None
                        state = 'idle'
                        relaxed_cnt = 0

            # Ghi frame GỐC (không overlay)
            if state == 'recording' and video_writer:
                video_writer.write(clean_frame)
                frame_count += 1

            # ══════════════════════════════════════════════
            # HEADER + STATE DISPLAY
            # ══════════════════════════════════════════════
            cv2.rectangle(frame, (0,0), (w,55), (30,30,30), -1)
            cv2.putText(frame, f"Nhan: {label_name.upper()}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Video: {video_count}", (10,48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame,
                        f"[A] Auto: {'ON' if auto_mode else 'OFF'}",
                        (w-250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0) if auto_mode else (100,100,100), 1)
            cv2.putText(frame,
                        f"[M] Mesh: {'ON' if show_mesh else 'OFF'}",
                        (w-250, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            if state == 'countdown':
                elapsed_cd = now - countdown_start
                num_color  = ((0,0,255) if elapsed_cd > self.COUNTDOWN_SECS - 2
                              else (0,165,255) if elapsed_cd > self.COUNTDOWN_SECS - 3
                              else (0,255,0))
                self._draw_countdown(frame, w, h, elapsed_cd, num_color)

            elif state == 'recording':
                self._draw_recording(frame, w, h,
                                     now - start_time, frame_count, relaxed_cnt)

            elif state == 'idle':
                in_cooldown = (now - last_stop_time) < self.COOLDOWN_SECS
                if in_cooldown and auto_mode:
                    cv2.putText(frame,
                                f"Cho {self.COOLDOWN_SECS-(now-last_stop_time):.1f}s ...",
                                (w//2-60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200,200,200), 1)
                elif auto_mode:
                    hint = "Gio tay len de bat dau" if relaxed else "Dieu chinh khung hinh..."
                    cv2.putText(frame, hint, (w//2-130, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
                else:
                    cv2.putText(frame, "[SPACE] de bat dau quay",
                                (w//2-120, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0,255,0), 1)

            # Relaxed indicator
            cv2.putText(frame,
                        "Tay: THA LONG" if relaxed else "Tay: GIO LEN",
                        (w-160, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100,100,255) if relaxed else (0,255,100), 1)

            # Footer
            cv2.rectangle(frame, (0,h-30), (w,h), (30,30,30), -1)
            cv2.putText(frame,
                        "[SPACE] Thu cong  |  [A] Auto  |  [M] Mesh  |  [Q] Thoat",
                        (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

            cv2.imshow('VSL Collector', frame)

            # ── Phím bấm ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if state != 'recording':
                    state = 'recording'
                    fp, video_writer, frame_count, start_time = \
                        self._start_recording_manual(
                            label_name, label_dir, video_count, fps, width, height)
                    relaxed_cnt = 0
                    print(f"  Thu cong: BAT DAU video {video_count+1}")
                else:
                    video_count, last_stop_time = self._stop_recording(
                        video_writer, video_count, frame_count,
                        time.time() - start_time, fp, label_name, time.time())
                    video_writer = None
                    state = 'idle'
                    relaxed_cnt = 0

            elif key in (ord('a'), ord('A')):
                auto_mode = not auto_mode
                if not auto_mode and state == 'countdown':
                    state = 'idle'
                print(f"  Auto: {'ON' if auto_mode else 'OFF'}")

            elif key in (ord('m'), ord('M')):
                show_mesh = not show_mesh

            elif key in (ord('q'), ord('Q')):
                if state == 'recording' and video_writer:
                    video_writer.release()
                    video_count += 1
                break

        cap.release()
        cv2.destroyAllWindows()
        self._ts = 0

        # Cập nhật metadata
        self.metadata['labels'][label_name] = dict(
            num_videos=video_count, path=label_dir)
        self.metadata['total_videos'] = sum(
            v['num_videos'] for v in self.metadata['labels'].values())
        self._save_meta()
        print(f"\n  Hoan thanh: {label_name} - {video_count} video")

    # ── Start / Stop helpers ──────────────────────────────

    def _make_video_path(self, label_name, label_dir, video_count):
        fn = f'{label_name}_{video_count:04d}.mp4'
        return os.path.join(label_dir, fn)

    def _start_recording(self, label_name, label_dir,
                          video_count, fps, width, height, now):
        fp = self._make_video_path(label_name, label_dir, video_count)
        vw = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
        return 'recording', fp, vw, 0, now

    def _start_recording_manual(self, label_name, label_dir,
                                 video_count, fps, width, height):
        fp = self._make_video_path(label_name, label_dir, video_count)
        vw = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
        return fp, vw, 0, time.time()

    def _stop_recording(self, video_writer, video_count,
                         frame_count, duration, fp, label_name, now):
        video_writer.release()
        video_count += 1
        print(f"  DUNG video {video_count} ({frame_count}f, {duration:.1f}s)")
        upload_to_hf(fp, label_name)
        return video_count, now

    # ══════════════════════════════════════════════════════
    # MENU
    # ══════════════════════════════════════════════════════

    def interactive_menu(self):
        while True:
            print("\n" + "="*60)
            print(" VSL COLLECTOR - MediaPipe Tasks API ".center(60, "="))
            print("="*60)
            print("  1. Xem thong ke video")
            print("  2. Tao nhan moi va thu video")
            print("  3. Tiep tuc thu video cho nhan co san")
            print("  4. Thu video IDLE (nghi / khong ky hieu)")
            print("  5. Upload video co san len HuggingFace")
            print("  6. Luu va thoat")
            print("="*60)

            ch = input("\n  Chon (1-6): ").strip()

            if ch == "1":
                self.show_statistics()

            elif ch == "2":
                lb = input("\n  Ten nhan moi (khong dau): ").strip().lower().replace(" ", "_")
                if not lb:
                    print("  Ten rong!"); continue
                viet = input(f"  Ten tieng Viet cho '{lb}': ").strip() or lb
                self._save_display_name(lb, viet)
                if lb in self.metadata['labels']:
                    if input(f"  '{lb}' da ton tai! Thu them? (y/n): ").strip().lower() != 'y':
                        continue
                self.collect_label(lb)

            elif ch == "3":
                labels = list(self.metadata['labels'].keys())
                if not labels:
                    print("  Chua co nhan nao!"); continue
                print("\n  Danh sach nhan:")
                for i, lb in enumerate(labels, 1):
                    n = self.metadata['labels'][lb]['num_videos']
                    print(f"  {i:>3}. {lb} ({n} video)")
                try:
                    idx = int(input("\n  Chon so: ").strip()) - 1
                    if 0 <= idx < len(labels):
                        self.collect_label(labels[idx])
                    else:
                        print("  Khong hop le!")
                except ValueError:
                    print("  Nhap so!")

            elif ch == "4":
                self._menu_idle()

            elif ch == "5":
                self._menu_upload_files()

            elif ch == "6":
                self._save_meta()
                self.show_statistics()
                print("\n  Tam biet!\n")
                break
            else:
                print("  Khong hop le!")

    def _menu_idle(self):
        """Sub-menu thu video IDLE."""
        idle_actions = [
            ("tay_xuoi_hong",     "Tay xuoi ben hong dung yen"),
            ("tay_khoanh_nguc",   "Tay khoanh truoc nguc"),
            ("tay_tren_ban",      "Tay dat tren ban"),
            ("ga_dau",            "Ga dau / chinh toc"),
            ("dua_tay_len_xuong", "Dua tay len roi ha xuong khong ky"),
            ("vuon_vai",          "Vuon vai / doi tu the"),
            ("dung_xa",           "Dung xa camera"),
            ("dung_gan",          "Dung gan camera"),
            ("nghieng_nguoi",     "Nghieng nguoi sang trai phai"),
            ("chi_tro",           "Chi tay ve phia truoc"),
            ("bo_tay_vao_tui",    "Bo tay vao tui quan"),
            ("voi_lay_do",        "Voi tay lay do"),
        ]

        print("\n" + "="*60)
        print(" THU VIDEO IDLE ".center(60))
        print("="*60)
        for i, (key, desc) in enumerate(idle_actions, 1):
            label    = f"__idle__{key}"
            existing = self.metadata['labels'].get(label, {}).get('num_videos', 0)
            status   = f"({existing} video)" if existing else "(chua co)"
            print(f"  {i:>2}. {desc:<40} {status}")
        print("   0. Thu tat ca theo thu tu")
        print("  99. Nhap ten hanh dong rieng")
        print("="*60)

        try:
            choice = input("\n  Chon (0 / 1-12 / 99): ").strip()

            if choice == "0":
                for key, desc in idle_actions:
                    label = f"__idle__{key}"
                    input(f"\n  Chuan bi: {desc}\n  Nhan ENTER de bat dau...")
                    self.collect_label(label)

            elif choice == "99":
                custom = input("  Ten hanh dong (vd: nhin_dien_thoai): ").strip()
                if not custom:
                    print("  Ten rong!"); return
                label = f"__idle__{custom}"
                viet  = input(f"  Ten tieng Viet cho '{label}': ").strip() or label
                self._save_display_name(label, viet)
                self.collect_label(label)

            else:
                idx = int(choice) - 1
                if 0 <= idx < len(idle_actions):
                    key, desc = idle_actions[idx]
                    label = f"__idle__{key}"
                    print(f"\n  Chuan bi: {desc}")
                    self.collect_label(label)
                else:
                    print("  Khong hop le!")

        except ValueError:
            print("  Nhap so!")

    # ══════════════════════════════════════════════════════
    # UPLOAD FILE CÓ SẴN LÊN HUGGINGFACE
    # ══════════════════════════════════════════════════════

    def _pick_files_gui(self) -> list[str]:
        """
        Mở hộp thoại chọn file bằng tkinter.
        Trả về list đường dẫn, hoặc [] nếu không có tkinter.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()          # ẩn cửa sổ tkinter chính
            root.attributes('-topmost', True)   # hiện hộp thoại lên trên
            paths = filedialog.askopenfilenames(
                title      = "Chon file video de upload",
                filetypes  = [
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All files",   "*.*"),
                ],
            )
            root.destroy()
            return list(paths)
        except Exception:
            return []   # fallback về nhập tay

    def _menu_upload_files(self):
        """
        Option 5: Chọn file MP4 từ máy → upload HuggingFace
                  → hỏi có muốn xử lý → .npy luôn không.
        """
        from collector.hf_upload import _hf_api, HF_REPO_ID

        print("\n" + "="*60)
        print(" UPLOAD VIDEO LEN HUGGINGFACE ".center(60))
        print("="*60)

        # ── Kiểm tra HF đã init chưa ──
        if _hf_api is None:
            print("\n  CANH BAO: HuggingFace chua duoc ket noi!")
            print("  Kiem tra file .env co HF_TOKEN va HF_REPO_ID chua.")
            input("\n  Nhan ENTER de quay lai menu...")
            return

        # ── Bước 1: Chọn file ──
        print("\n  Chon file bang:")
        print("  [1] Hop thoai chon file (GUI)")
        print("  [2] Nhap duong dan thu cong")
        ch = input("\n  Chon (1/2): ").strip()

        selected_files = []

        if ch == "1":
            print("\n  Dang mo hop thoai chon file...")
            selected_files = self._pick_files_gui()
            if not selected_files:
                print("  (Khong co tkinter hoac khong chon file nao)")
                print("  Chuyen sang nhap tay...")
                ch = "2"   # fallback

        if ch == "2":
            print("\n  Nhap duong dan file (1 dong 1 file, dong trong de ket thuc):")
            while True:
                p = input("  > ").strip().strip('"').strip("'")
                if not p:
                    break
                if os.path.isfile(p):
                    selected_files.append(p)
                    print(f"    ✓ Da them: {os.path.basename(p)}")
                elif os.path.isdir(p):
                    # Nếu nhập thư mục → lấy tất cả video trong đó
                    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                    found = [os.path.join(p, f) for f in sorted(os.listdir(p))
                             if os.path.splitext(f)[1].lower() in exts]
                    selected_files.extend(found)
                    print(f"    ✓ Da them {len(found)} video tu thu muc")
                else:
                    print(f"    ✗ Khong tim thay: {p}")

        if not selected_files:
            print("\n  Khong co file nao duoc chon.")
            input("  Nhan ENTER de quay lai...")
            return

        # ── Bước 2: Xác nhận danh sách ──
        print(f"\n  Da chon {len(selected_files)} file:")
        for i, fp in enumerate(selected_files, 1):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  {i:>3}. {os.path.basename(fp):<45} {size_mb:.1f} MB")

        # ── Bước 3: Nhập label ──
        print("\n  Nhap label cho cac video nay.")
        print("  (Cac file se duoc upload vao videos/<label>/)")

        # Gợi ý label đang có
        if self.metadata['labels']:
            labels_exist = list(self.metadata['labels'].keys())
            print("\n  Label hien co:")
            for i, lb in enumerate(labels_exist, 1):
                print(f"    {i}. {lb}")
            print("\n  Nhap ten label moi hoac so thu tu label co san:")
            ans = input("  > ").strip()
            try:
                idx = int(ans) - 1
                if 0 <= idx < len(labels_exist):
                    label_name = labels_exist[idx]
                else:
                    label_name = ans.lower().replace(" ", "_")
            except ValueError:
                label_name = ans.lower().replace(" ", "_")
        else:
            label_name = input("  Ten label: ").strip().lower().replace(" ", "_")

        if not label_name:
            print("  Ten label rong! Huy."); return

        # Hỏi tên tiếng Việt nếu label mới
        if label_name not in self.metadata.get('labels', {}):
            viet = input(f"  Ten tieng Viet cho '{label_name}': ").strip()
            if viet:
                self._save_display_name(label_name, viet)

        # ── Bước 4: Xác nhận upload ──
        print(f"\n  Se upload {len(selected_files)} file vao:")
        print(f"  HF repo : {HF_REPO_ID}")
        print(f"  Path    : videos/{label_name}/")

        confirm = input("\n  Xac nhan upload? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  Da huy."); return

        # ── Bước 5: Upload từng file ──
        print(f"\n  Bat dau upload...")
        success = 0
        failed  = []

        # Copy file vào data/videos/<label>/ để đồng bộ với metadata
        label_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        for i, fp in enumerate(selected_files, 1):
            fname = os.path.basename(fp)
            print(f"  [{i}/{len(selected_files)}] {fname}...", end=" ", flush=True)

            # Copy vào thư mục local nếu file nằm ngoài
            local_target = os.path.join(label_dir, fname)
            if os.path.abspath(fp) != os.path.abspath(local_target):
                import shutil
                shutil.copy2(fp, local_target)

            ok = upload_to_hf(local_target, label_name)
            if ok:
                print("✓")
                success += 1
            else:
                print("✗ (loi)")
                failed.append(fname)

        # ── Bước 6: Cập nhật metadata ──
        existing_count = self.metadata['labels'].get(label_name, {}).get('num_videos', 0)
        self.metadata['labels'][label_name] = dict(
            num_videos = existing_count + success,
            path       = label_dir,
        )
        self.metadata['total_videos'] = sum(
            v['num_videos'] for v in self.metadata['labels'].values())
        self._save_meta()

        # ── Bước 7: Tổng kết ──
        print(f"\n  Ket qua: {success}/{len(selected_files)} file upload thanh cong")
        if failed:
            print(f"  That bai ({len(failed)} file):")
            for f in failed:
                print(f"    - {f}")

        # ── Bước 8: Hỏi có muốn xử lý → .npy không ──
        if success > 0:
            print("\n" + "-"*50)
            ans = input(
                "  Xu ly cac video nay -> .npy de train luon? (y/n): "
            ).strip().lower()
            if ans == 'y':
                self._process_uploaded_to_npy(label_dir, label_name)

        input("\n  Nhan ENTER de quay lai menu...")

    def _process_uploaded_to_npy(self, video_dir: str, label_name: str):
        """Gọi video_to_npy pipeline ngay sau khi upload."""
        try:
            # Import converter (cần video_to_npy.py + converter/ cùng cấp)
            from converter import KeypointNormalizer, resample_sequence, Augmenter
            from vsl.extractor import VideoExtractor
            from vsl.config    import cfg as vsl_cfg
            import numpy as np
        except ImportError as e:
            print(f"\n  Khong the import converter: {e}")
            print("  Hay chay video_to_npy.py rieng de xu ly.")
            return

        print(f"\n  Bat dau xu ly video -> .npy cho label '{label_name}'...")

        ext_ok  = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos  = sorted([f for f in os.listdir(video_dir)
                          if os.path.splitext(f)[1].lower() in ext_ok])
        if not videos:
            print("  Khong tim thay video."); return

        extractor  = VideoExtractor()
        augmenter  = Augmenter()
        output_dir = os.path.join('data', 'processed', label_name)
        os.makedirs(output_dir, exist_ok=True)

        success = 0
        for i, vf in enumerate(videos, 1):
            vpath = os.path.join(video_dir, vf)
            print(f"  [{i}/{len(videos)}] {vf}")

            import cv2
            cap = cv2.VideoCapture(vpath)
            raw = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feats, _ = extractor.extract_frame(rgb)
                feats    = KeypointNormalizer.normalize_frame(feats)
                raw.append(feats)
            cap.release()

            if len(raw) < 5:
                print(f"    CANH BAO: Qua ngan ({len(raw)} frames). Bo qua.")
                continue

            normalized  = resample_sequence(raw, vsl_cfg.SEQ_LEN)
            vid_id      = os.path.splitext(vf)[0]
            augs        = augmenter.generate(normalized)
            for suffix, data in augs:
                fn   = f"{vid_id}_{suffix}.npy"
                path = os.path.join(output_dir, fn)
                np.save(path, data.astype(np.float32))
            print(f"    → {len(augs)} file .npy da luu vao {output_dir}/")
            success += 1

        extractor.close()
        print(f"\n  Hoan thanh: {success}/{len(videos)} video da xu ly.")
        print(f"  File .npy tai: data/processed/{label_name}/")

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    collector = WebcamVideoCollector(output_dir='data/videos')
    try:
        collector.interactive_menu()
    finally:
        collector.close()


if __name__ == "__main__":
    main()