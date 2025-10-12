import cv2
import os
import time
import threading
from datetime import datetime
import psutil
import mediapipe as mp
from queue import Queue
import argparse
from smol_query import SmolVLMQuery  # Make sure smol_query.py is in same folder

# --- Config ---
OUT_DIR = "frames"
LOG_FILE = "gesture_logs.txt"
FPS_TARGET = 5.0  # Smooth webcam
SMOL_PROMPT = "Classify this hand gesture: Thumbs Up, Thumbs Down, 1 Finger, 2 Fingers, 3 Fingers, 4 Fingers, 5 Fingers, Unknown"

# --- Directories ---
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=2,
                                min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- SmolVLM setup ---
smol_query = SmolVLMQuery(endpoint="http://localhost:5000/v1")  # Replace with hosted API if available

# --- Thread-safe queue ---
task_queue = Queue()

# --- Utilities ---
def write_log(ts, hand, finger_count, final_gesture, cpu, mem):
    sep = "-"*50 + "\n"
    with open(LOG_FILE, "a") as f:
        f.write(
            f"Timestamp: {ts}\n"
            f"Hand Side: {hand}\n"
            f"FingerCount: {finger_count}\n"
            f"Final Gesture: {final_gesture}\n"
            f"CPU: {cpu}%\n"
            f"MEM: {mem} MB\n"
            f"{sep}"
        )

def save_frame(frame, hand_side, gesture):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{hand_side}_{gesture}_{ts}.jpg"
    cv2.imwrite(os.path.join(OUT_DIR, fname), frame)

# --- Finger counting ---
def count_fingers(hand_landmarks, handedness):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if handedness == "Right":
        fingers.append(hand_landmarks.landmark[tips[0]].x <
                       hand_landmarks.landmark[tips[0]-1].x)
    else:
        fingers.append(hand_landmarks.landmark[tips[0]].x >
                       hand_landmarks.landmark[tips[0]-1].x)

    # Other 4 fingers
    for tip in tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y <
                       hand_landmarks.landmark[tip-2].y)
    return fingers.count(True), fingers

# --- Detect Thumbs Up / Down ---
def detect_thumb_gesture(hand_landmarks, fingers):
    if fingers[0] and sum(fingers[1:]) == 0:
        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y:
            return "Thumbs Up"
        elif hand_landmarks.landmark[4].y > hand_landmarks.landmark[2].y:
            return "Thumbs Down"
    return None

# --- Worker thread ---
def worker():
    while True:
        item = task_queue.get()
        if item is None:
            break
        frame, hand_side, finger_count, final_gesture = item

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cpu = psutil.cpu_percent(interval=None)
        mem = int(psutil.virtual_memory().used / (1024*1024))

        write_log(ts, hand_side, finger_count, final_gesture, cpu, mem)
        save_frame(frame, hand_side, final_gesture)

        cv2.putText(frame, f"{hand_side}: {final_gesture}",
                    (10, 30 if hand_side.startswith("Left") else 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        task_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# --- SmolVLM async batch query ---
def query_smol_batch(hands):
    """
    hands = list of dicts: {'crop': img, 'hand_side': str, 'finger_count': int}
    """
    crops = [h['crop'] for h in hands]
    try:
        results = smol_query.query_batch(crops, SMOL_PROMPT)
    except Exception as e:
        print("SmolVLM error:", e)
        results = ["Unknown"] * len(hands)

    # Push results to worker queue
    for h, r in zip(hands, results):
        task_queue.put((h['crop'], h['hand_side'], h['finger_count'], r))

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='mediapipe', choices=['mediapipe','smolvlm','hybrid'])
args = parser.parse_args()
mode = args.mode

# --- Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    now = time.time()
    elapsed = now - prev_time

    smol_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label
            hand_side = "Left Hand" if label == "Right" else "Right Hand"

            finger_count, finger_list = count_fingers(hand_landmarks, label)
            local_gesture = f"{finger_count} Fingers" if finger_count > 0 else "Unknown"
            thumb_gesture = detect_thumb_gesture(hand_landmarks, finger_list)
            final_gesture = thumb_gesture if thumb_gesture else local_gesture

            # Overlay temporary gesture
            cv2.putText(frame, f"{hand_side}: {final_gesture}",
                        (10, 30 if hand_side.startswith("Left") else 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Collect for SmolVLM batch
            if mode in ["smolvlm", "hybrid"]:
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark])*w) - 20
                x_max = int(max([lm.x for lm in hand_landmarks.landmark])*w) + 20
                y_min = int(min([lm.y for lm in hand_landmarks.landmark])*h) - 20
                y_max = int(max([lm.y for lm in hand_landmarks.landmark])*h) + 20
                crop = frame[max(0,y_min):min(h,y_max), max(0,x_min):min(w,x_max)]

                smol_hands.append({'crop': crop, 'hand_side': hand_side, 'finger_count': finger_count})

                if mode == "smolvlm":
                    final_gesture = "Detecting..."  # placeholder until SmolVLM returns

    # Run batch query asynchronously
    if smol_hands:
        threading.Thread(target=query_smol_batch, args=(smol_hands,), daemon=True).start()

    else:
        cv2.putText(frame, "No Hand Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
task_queue.put(None)
cap.release()
cv2.destroyAllWindows()
