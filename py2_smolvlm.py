#!/usr/bin/env python3
"""
py2_smolvlm.py — Updated for Engine Integration
-------------------------------------------------------
SmolVLM + MediaPipe hybrid (Config-aware version)
• Reads parameters from config.json
• Detects both left & right hands
• Draws clean skeleton and saves crops
• Uses SmolVLM for gesture reasoning
• Logs outputs with timestamps
-------------------------------------------------------
"""

import os
import cv2
import time
import torch
import json
import threading
import traceback
import mediapipe as mp
from datetime import datetime
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- GLOBALS ----------------
DEVICE = "cpu"
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT = (
    "<image>\nIn one short line: Hand side (Left or Right); Gesture; Fingers raised (0-5). "
    "Gesture options: Thumbs Up, Thumbs Down, Open Hand, Closed Fist, Other. "
    "Format exactly: Side; Gesture; Count"
)

latest_text_left, latest_text_right = "Detecting...", "Detecting..."
lock = threading.Lock()
last_infer_time = {"Left": 0.0, "Right": 0.0}
threads = {"Left": None, "Right": None}

# ---------------- FUNCTIONS ----------------
def load_model():
    """Load the SmolVLM model."""
    print("[INFO] Loading SmolVLM model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    print(f"[READY] SmolVLM loaded on {DEVICE}")
    return processor, model


def run_smolvlm(crop, hand_side, processor, model, out_dir, log_file):
    """Run SmolVLM reasoning on a cropped hand region."""
    global latest_text_left, latest_text_right
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[rgb], text=[PROMPT], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=20)
        raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        text = raw.split("\n")[0]

        # Save result to globals
        with lock:
            if hand_side == "Left":
                latest_text_left = text
            else:
                latest_text_right = text

        # Save image with label
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{hand_side}_{ts}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), crop)

        # Log
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {hand_side} => {text}\n")

        print(f"🧠 {hand_side} => {text}")

    except Exception as e:
        print(f"[ERROR] SmolVLM reasoning failed: {e}")
        traceback.print_exc()


def main(config=None):
    """Main entrypoint — runs SmolVLM + Mediapipe gesture detection."""
    try:
        # -------------- CONFIG --------------
        base_dir = os.path.dirname(__file__)
        out_dir = os.path.join(base_dir, "frames_smolvlm")
        log_file = os.path.join(base_dir, "smolvlm_log.txt")
        os.makedirs(out_dir, exist_ok=True)

        use_webcam = True
        fps_target = 5
        latency_limit_ms = 250

        if config:
            use_webcam = config.get("use_webcam", True)
            fps_target = config.get("cfg_metrics", {}).get("fps_target", 5)
            latency_limit_ms = config.get("cfg_metrics", {}).get("latency_limit_ms", 250)

        print(f"[CONFIG] Webcam: {use_webcam} | FPS: {fps_target} | Latency: {latency_limit_ms}ms")

        processor, model = load_model()

        # -------------- VIDEO --------------
        cap = cv2.VideoCapture(0 if use_webcam else config.get("video_path", ""))
        if not cap.isOpened():
            print("❌ Could not open video source.")
            return

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        print("🎥 Running SmolVLM + MediaPipe — Press 'q' to quit.")

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        ) as hands:
            prev_time = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                h, w, _ = frame.shape

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        side = handedness.classification[0].label
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )

                        # Crop hand region
                        x_min, y_min, x_max, y_max = w, h, 0, 0
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, y_min = min(x_min, x), min(y_min, y)
                            x_max, y_max = max(x_max, x), max(y_max, y)
                        pad = 40
                        crop = frame[max(0, y_min - pad):min(h, y_max + pad),
                                     max(0, x_min - pad):min(w, x_max + pad)]

                        # Run reasoning every latency window
                        now = time.time()
                        if now - last_infer_time[side] >= (latency_limit_ms / 1000.0):
                            if crop.size > 0 and (threads[side] is None or not threads[side].is_alive()):
                                threads[side] = threading.Thread(
                                    target=run_smolvlm,
                                    args=(crop, side, processor, model, out_dir, log_file),
                                    daemon=True
                                )
                                threads[side].start()
                                last_infer_time[side] = now

                # Overlay results
                with lock:
                    cv2.putText(frame, f"Left:  {latest_text_left}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right: {latest_text_right}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # FPS counter
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow("SmolVLM + Dual-Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ SmolVLM analysis complete.")
        print(f"Frames saved to: {out_dir}")
        print(f"Logs saved to: {log_file}")

    except Exception as e:
        print(f"[FATAL] Error in SmolVLM main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
