# Gesture Detection using OpenCV and SmolVLM

## 📌 Project Overview
This project implements a **gesture detection system** using:
- **OpenCV + MediaPipe** (for hand tracking and gesture classification)
- **SmolVLM** (as an alternative detection module)

It benchmarks and compares both approaches in terms of **CPU usage, memory usage, and FPS performance**.  
The system detects hand presence, identifies hand side (Left/Right), counts fingers, and recognizes gestures like **Thumbs Up/Down**.

---

## ✨ Features
- ✅ **Hand Detection** → Detects if a hand is present and its bounding box  
- ✅ **Hand Side Identification** → Left Hand / Right Hand  
- ✅ **Finger Counting** → Counts 0–5 open fingers  
- ✅ **Gesture Recognition**:
  - 👍 Thumbs Up
  - 👎 Thumbs Down
  - ✊ Hand Down (Fist)
  - ☝ Index Finger
  - ✌ Middle Finger
  - 🤟 Ring Finger
  - 🤘 Little Finger
  - 🖐 Open Hand
- ✅ **Both Hands Detection** (simultaneously)  
- ✅ **Logging** → CPU, memory usage, FPS, gesture results  
- ✅ **Benchmarking** → Compare OpenCV vs SmolVLM  

---

## 📂 Folder Structure
```
gesture_submission_ready/
│── opencv/
│   ├── gesture_opencv.py
│   ├── gesture_outputs.log
│   ├── frames/ (saved gesture snapshots)
│   ├── architecture_diagram.pdf
│   └── dfd_diagram.pdf
│
│── smolvlm/
│   ├── gesture_smolvlm.py
│   ├── gesture_outputs.log
│   ├── frames/
│   ├── architecture_diagram.pdf
│   └── dfd_diagram.pdf
│
│── benchmarks/
│   ├── run_benchmarks.py
│   └── results_summary.csv
│
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/Nithinnithu0610/Gesture_detection-using-opencv-and-smolVLM-method.git
cd Gesture_detection-using-opencv-and-smolVLM-method
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run OpenCV Gesture Detection
```bash
python opencv/gesture_opencv.py --out_dir opencv/frames --log opencv/gesture_outputs.log
```

### Run SmolVLM Gesture Detection
```bash
python smolvlm/gesture_smolvlm.py --out_dir smolvlm/frames --log smolvlm/gesture_outputs.log
```

### Run Benchmarks (20 seconds)
```bash
python benchmarks/run_benchmarks.py 20
```

---

## 📤 Outputs

1. **Gesture Logs**  
   - `opencv/gesture_outputs.log`  
   - `smolvlm/gesture_outputs.log`  
   Format:
   ```
   Timestamp: 2025-09-29 15:20:01
   Hand Presence: Hand Detected
   Hand Side: Left Hand
   Finger Count: 1
   Gesture Detected: Thumbs Up
   CPU Usage: 25%
   Memory Usage: 150 MB
   Frames Processed Per Second: 28
   -------------------------------------------------
   ```

2. **Saved Frames**  
   Snapshots stored in:
   - `opencv/frames/`
   - `smolvlm/frames/`

3. **Benchmarks Summary**  
   - `benchmarks/results_summary.csv`
   ```
   module,cpu_mean,mem_mean_mb,fps_mean
   opencv,27.5,155.3,29
   smolvlm,33.2,182.1,23
   ```

---

## 📋 Requirements
- Python 3.10+
- OpenCV
- MediaPipe
- psutil
- pillow
- numpy

Install them via:
```bash
pip install -r requirements.txt
```

---

## 📝 Notes
- Press **Q** to quit gesture detection.  
- Ensure your **camera is free** before running benchmarks.  
- Logs and frames will be saved automatically.  

---

## 👨‍💻 Author
**Nithin**  
GitHub: [Nithinnithu0610](https://github.com/Nithinnithu0610)  
