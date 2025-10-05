import cv2
import time
import os
import psutil
import threading
import queue
from smol_query import SmolVLMQuery

# --- Directories ---
OUTPUT_DIR = "outputs/frames"
LOG_DIR = "outputs/logs"
LOG_FILE = os.path.join(LOG_DIR, "gesture_outputs.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Globals ---
answer = "Waiting..."
frame_queue = queue.Queue(maxsize=1)   # only keep latest frame
INFERENCE_INTERVAL = 3                 # seconds between inferences
counter = 0

# --- Load Model ---
print("Loading SmolVLM model on CPU (may take 1-2 min)...")
smol = SmolVLMQuery()  # if GPU available: SmolVLMQuery(device="cuda")
print("SmolVLM loaded.")

# --- Logging (multi-line style) ---
def save_log(timestamp, gesture, fps):
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().used // (1024 * 1024)
    with open(LOG_FILE, "a") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Gesture Detected: {gesture}\n")
        f.write(f"CPU Usage: {cpu}%\n")
        f.write(f"Memory Usage: {mem} MB\n")
        f.write(f"Frames Processed Per Second: {fps:.0f}\n")
        f.write("-" * 50 + "\n")

# --- Worker Thread ---
def inference_worker():
    global answer, counter
    last_inference_time = 0
    while True:
        try:
            frame = frame_queue.get()   # waits for a frame
            now = time.time()
            if now - last_inference_time < INFERENCE_INTERVAL:
                continue  # skip until interval passes

            last_inference_time = now
            counter += 1

            start = time.time()
            small = cv2.resize(frame, (128, 128))
            ans = smol.query(
                small,
                "<image>\nHow many hands are visible? For each hand, is it left or right, and how many fingers are up?"
            )
            answer = ans
            fps = 1.0 / max(time.time() - start, 0.001)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save every frame used for inference
            frame_file = os.path.join(
                OUTPUT_DIR,
                f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(frame_file, frame)

            save_log(timestamp, answer, fps)
            print(f"[{timestamp}] {answer} | FPS: {fps:.1f}")

        except Exception as e:
            print("Inference error:", e)

# Start worker thread
threading.Thread(target=inference_worker, daemon=True).start()

# --- Camera Setup ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend faster on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Push frame to queue (only latest kept)
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    # Show display with latest answer
    display_frame = frame.copy()
    cv2.putText(display_frame, answer, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Gesture Detection - SmolVLM", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
