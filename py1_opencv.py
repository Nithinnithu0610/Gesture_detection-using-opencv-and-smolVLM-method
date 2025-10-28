#!/usr/bin/env python3
"""
Static Approach — OpenCV + MediaPipe (Dual-Hand Accurate Finger Counting)
---------------------------------------------------------------------------
Detects hand landmarks, counts fingers, recognizes gestures,
reads parameters from config.json, logs details in a readable format,
and saves all gesture frames in a unified folder.
"""

import os
import sys
import cv2
import time
import psutil
from datetime import datetime
import mediapipe as mp
import json

# -------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")  # ← FIXED PATH

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

OUT_DIR = os.path.join(BASE_DIR, "frames_opencv_static")
LOG_FILE = os.path.join(BASE_DIR, "opencv_static_log.txt")
os.makedirs(OUT_DIR, exist_ok=True)

use_webcam = cfg.get("use_webcam", True)
video_path = cfg.get("video_path", "")
expected_fps = cfg.get("cfg_metrics", {}).get("fps_target", 10)
latency_ms = cfg.get("cfg_metrics", {}).get("latency_limit_ms", 200)

# -------------------------------------------------------
# MediaPipe Setup
# -------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def count_fingers(hand_landmarks, handedness: str):
    """Count fingers accurately, adjusted for mirrored webcam."""
    tips = [4, 8, 12, 16, 20]
    fingers_bool = []

    # For index, middle, ring, pinky
    for tip in tips[1:]:
        fingers_bool.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    finger_count = sum(fingers_bool)

    # --- FIXED Thumb handling (no +1 overcount) ---
    palm_x = (hand_landmarks.landmark[5].x + hand_landmarks.landmark[17].x) / 2
    thumb_x = hand_landmarks.landmark[4].x
    wrist_y = hand_landmarks.landmark[0].y
    thumb_tip_y = hand_landmarks.landmark[4].y

    if handedness == "Right":
        thumb_clear = thumb_x < palm_x - 0.05
        thumb_up = thumb_tip_y < wrist_y - 0.07
        thumb_down = thumb_tip_y > wrist_y + 0.07
    else:
        thumb_clear = thumb_x > palm_x + 0.05
        thumb_up = thumb_tip_y < wrist_y - 0.07
        thumb_down = thumb_tip_y > wrist_y + 0.07

    if thumb_clear:
        finger_count += 1

    return finger_count, thumb_up, thumb_down


def detect_gesture(finger_count, thumb_up, thumb_down):
    """Map finger counts to gesture names."""
    if thumb_up and finger_count == 1:
        return "Thumbs_Up"
    if thumb_down and finger_count == 1:
        return "Thumbs_Down"

    gestures = {
        0: "Fist",
        1: "One_Finger",
        2: "Two_Fingers",
        3: "Three_Fingers",
        4: "Four_Fingers",
        5: "Open_Hand",
    }
    return gestures.get(finger_count, f"{finger_count}_Fingers")


def write_log(ts, side, gesture, count, cpu, mem, fps):
    """Write detailed multi-line log for each frame."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("--------------------------------------------------\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Hand Presence: {'No Hand' if gesture == 'No_Hand' else 'Yes'}\n")
        f.write(f"Hand Side: {side}\n")
        f.write(f"Finger Count: {count}\n")
        f.write(f"Gesture Detected: {gesture}\n")
        f.write(f"CPU Usage: {cpu:.1f}%\n")
        f.write(f"Memory Usage: {mem} MB\n")
        f.write(f"Frames Processed Per Second: {fps:.1f}\n")
        f.write("--------------------------------------------------\n")

# -------------------------------------------------------
# Main Function
# -------------------------------------------------------
def main(_cfg=None):
    cap = cv2.VideoCapture(0 if use_webcam else video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video source: {'Webcam' if use_webcam else video_path}")

    print("=== Dual-Hand Gesture Detection (OpenCV + MediaPipe) ===")
    print(f"[CONFIG] Webcam: {use_webcam} | FPS Target: {expected_fps} | Latency: {latency_ms}ms")
    print("[INFO] Press 'q' to quit.\n")

    prev_time = time.time()
    smooth_fps = expected_fps
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of stream or camera error.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        curr_time = time.time()
        inst_fps = 1.0 / max(1e-6, curr_time - prev_time)
        smooth_fps = 0.9 * smooth_fps + 0.1 * inst_fps
        prev_time = curr_time

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                side = hd.classification[0].label
                finger_count, thumb_up, thumb_down = count_fingers(lm, side)
                gesture = detect_gesture(finger_count, thumb_up, thumb_down)

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"{side}: {gesture} ({finger_count})",
                            (10, 40 if side == "Right" else 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if side == "Right" else (0, 255, 255), 2)

                # Save frame periodically
                if (time.time() - last_capture_time) > (latency_ms / 1000.0):
                    fname = f"{side}_{gesture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(OUT_DIR, fname), frame)
                    last_capture_time = time.time()

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu = psutil.cpu_percent()
                mem = int(psutil.virtual_memory().used / (1024 * 1024))
                write_log(ts, side, gesture, finger_count, cpu, mem, smooth_fps)

        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Dual-Hand Gesture Detection — OpenCV + MediaPipe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Static Approach completed successfully.")
    print(f"Logs saved to: {LOG_FILE}")
    print(f"Frames saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
