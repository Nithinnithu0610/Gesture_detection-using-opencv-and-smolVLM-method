# Gesture Detection using OpenCV + MediaPipe and SmolVLM

## 📌 Project Overview

This project implements a **real-time gesture detection system** using:

* **OpenCV + MediaPipe** for hand tracking
* **SmolVLM** for optional gesture classification

It captures hand presence, identifies hand side (Left/Right), counts fingers, and recognizes gestures like **Thumbs Up/Down**, **1–5 Fingers**, and more.
SmolVLM can be used as a classification module while MediaPipe ensures fast webcam performance.

---

## ✨ Features

* ✅ **Hand Detection** – Detects hand presence
* ✅ **Hand Side Identification** – Left Hand / Right Hand
* ✅ **Finger Counting** – Counts 0–5 fingers
* ✅ **Gesture Recognition**:

  * 👍 Thumbs Up
  * 👎 Thumbs Down
  * ✊ Fist
  * ☝ Index Finger
  * ✌ Two Fingers
  * 🖐 Open Hand
* ✅ **Hybrid Mode** – MediaPipe detects hands, SmolVLM optionally classifies gestures
* ✅ **Logging** – CPU, memory usage, gesture results, timestamps
* ✅ **Frame Capture** – Saves snapshots of detected gestures

---

## 🐂 Folder Structure

```
Gesture_Recognition/
├── gesture_main.py          # Main script, supports --mode argument
├── smol_query.py            # SmolVLM query wrapper
├── frames/                  # Saved gesture frames
├── gesture_logs.txt         # Logs CPU, memory, gestures
├── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nithinnithu0610/Gesture_detection-using-opencv-and-smolVLM-method.git
cd Gesture_detection-using-opencv-and-smolVLM-method
```

### 2. Create and activate virtual environment

```bash
python -m venv smolvlm_env
source smolvlm_env/Scripts/activate  # Windows bash
# or
smolvlm_env\Scripts\activate         # Windows CMD/PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run in MediaPipe only (fast webcam detection)

```bash
python gesture_main.py --mode mediapipe
```

### Run in SmolVLM only (requires endpoint)

```bash
python gesture_main.py --mode smolvlm
```

### Run in Hybrid mode (MediaPipe + SmolVLM)

```bash
python gesture_main.py --mode hybrid
```

Press **Q** to quit the webcam window.

---

## 📤 Outputs

1. **Gesture Logs** – `gesture_logs.txt`
   Example:

```
Timestamp: 2025-10-12_14-52-10
Hand Side: Right Hand
FingerCount: 3
Final Gesture: Thumbs Up
CPU: 36.5%
MEM: 7233 MB
```

2. **Saved Frames** – `frames/` folder, each frame is labeled with hand side and gesture.

---

## 📋 Requirements

* Python 3.10+
* OpenCV
* MediaPipe
* psutil
* requests
* numpy
* pillow

Install via:

```bash
pip install -r requirements.txt
```

---

## 📕 Notes

* SmolVLM classification requires a **running backend** or hosted API.
* MediaPipe acts as a **fallback** if SmolVLM is unavailable.
* Adjust `FPS_TARGET` in `gesture_main.py` for performance tuning.

---

## 👨‍💼 Author

**Nithin**
GitHub: [Nithinnithu0610](https://github.com/Nithinnithu0610)
