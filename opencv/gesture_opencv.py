#!/usr/bin/env python3
import cv2, time, os, argparse, psutil
import mediapipe as mp
from datetime import datetime

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(lm, handedness):
    """Return finger states: [thumb, index, middle, ring, pinky]"""
    fingers = []
    # Thumb check is mirrored for right vs left hand
    if handedness == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    # Other fingers: tip higher than pip → extended
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip].y < lm[tip-2].y else 0)
    return fingers

def classify_gesture(fingers, lm, handedness):
    """Classify gesture from finger states and landmarks"""
    thumb, index, middle, ring, pinky = fingers
    count = sum(fingers)

    # Thumbs Up/Down when only thumb is up
    if count == 1 and thumb == 1:
        if lm[4].y < lm[3].y:  # tip higher → up
            return f"{handedness} Thumbs Up"
        else:
            return f"{handedness} Thumbs Down"

    if count == 0:
        return f"{handedness} Hand Down"
    elif count == 1 and index == 1:
        return f"{handedness} Index Finger"
    elif count == 2 and index and middle:
        return f"{handedness} Two Fingers"
    elif count == 3:
        return f"{handedness} Three Fingers"
    elif count == 4:
        return f"{handedness} Four Fingers"
    elif count == 5:
        return f"{handedness} Open Hand"
    else:
        return f"{handedness} {count} Fingers"

def write_log(logpath, timestamp, hand_presence, hand_side, finger_count, gesture, cpu, mem_mb, fps):
    sep = "-"*49 + "\n"
    with open(logpath, "a") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Hand Presence: {hand_presence}\n")
        f.write(f"Hand Side: {hand_side}\n")
        f.write(f"Finger Count: {finger_count}\n")
        f.write(f"Gesture Detected: {gesture}\n")
        f.write(f"CPU Usage: {cpu}%\n")
        f.write(f"Memory Usage: {mem_mb} MB\n")
        f.write(f"Frames Processed Per Second: {fps}\n")
        f.write(sep)

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def main(args):
    ensure_dirs(args.out_dir)
    ensure_dirs(os.path.dirname(args.log))
    cap = cv2.VideoCapture(0 if not args.input else args.input)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)
    saved_frames = {}
    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            now = time.time()
            fps = round(1.0 / (now - prev_time), 2) if (now - prev_time) > 0 else 0.0
            prev_time = now

            hand_presence = "No Hand"
            if res.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_label = hand_handedness.classification[0].label
                    lm = hand_landmarks.landmark

                    fingers = fingers_up(lm, hand_label)
                    finger_count = sum(fingers)
                    gesture_name = classify_gesture(fingers, lm, hand_label)

                    hand_presence = "Hand Detected"
                    hand_side = f"{hand_label} Hand"

                    # Save sample frames (up to N per gesture)
                    key = f"{hand_label}_{gesture_name}"
                    if key not in saved_frames:
                        saved_frames[key] = 0
                    if saved_frames[key] < args.max_frames_per_gesture:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = os.path.join(args.out_dir, f"{hand_label}_{gesture_name}_{ts}_{frame_idx}.jpg")
                        cv2.imwrite(fname, frame)
                        saved_frames[key] += 1

                    # System metrics
                    cpu = psutil.cpu_percent(interval=None)
                    mem_bytes = psutil.virtual_memory().used
                    mem_mb = int(mem_bytes / (1024*1024))

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    write_log(args.log, timestamp, hand_presence, hand_side, finger_count, gesture_name, cpu, mem_mb, fps)

                    # Draw gesture label
                    cv2.putText(frame, f"{hand_label}: {gesture_name}", (10, 30 if hand_label=='Right' else 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                # Log "no hand" periodically
                if frame_idx % 30 == 0:
                    cpu = psutil.cpu_percent(interval=None)
                    mem_bytes = psutil.virtual_memory().used
                    mem_mb = int(mem_bytes / (1024*1024))
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    write_log(args.log, timestamp, "No Hand", "N/A", 0, "No Gesture", cpu, mem_mb, fps)

            cv2.imshow("Gesture Recognition - OpenCV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='', help='video path or blank for webcam')
    parser.add_argument('--out_dir', default='opencv/frames', help='where to save frames')
    parser.add_argument('--log', default='opencv/gesture_outputs.log', help='log file path')
    parser.add_argument('--max_frames_per_gesture', type=int, default=3)
    args = parser.parse_args()
    main(args)
