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
Gesture_Recognition_unzipped/
├── Gesture_Recognition/
│   ├── README.md
│   ├── requirements.txt
│   ├── benchmarks/
│   │   ├── results_summary.csv
│   │   ├── run_benchmarks.py
│   ├── opencv/
│   │   ├── architecture.md
│   │   ├── architecture_diagram.pdf
│   │   ├── dfd.md
│   │   ├── dfd_diagram.pdf
│   │   ├── gesture_opencv.py
│   │   ├── gesture_outputs.log
│   │   ├── gutputs.log
│   │   ├── frames_open cv/
│   │   │   ├── .gitkeep
│   │   │   ├── Left_Hand Down_20250929_182828_348.jpg
│   │   │   ├── Left_Index Finger_20250929_182828_347.jpg
│   │   │   ├── Left_Left 2 Fingers_20250929_190159_113.jpg
│   │   │   ├── Left_Left Four Fingers_20250929_190751_92.jpg
│   │   │   └── ... (many more gesture images)
│   ├── smolvlm/
│   │   ├── architecture.md
│   │   ├── architecture_diagram.pdf
│   │   ├── dfd.md
│   │   ├── dfd_diagram.pdf
│   │   ├── gesture_smolvlm.py
│   │   ├── gesture_outputs.log
│   │   ├── frames_smolvlm/
│   │   │   ├── .gitkeep
│   │   │   └── ... (gesture frames)
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
python opencv/gesture_opencv.py --out_dir opencv/frames_open\ cv --log opencv/gesture_outputs.log
```

### Run SmolVLM Gesture Detection
```bash
python smolvlm/gesture_smolvlm.py --out_dir smolvlm/frames_smolvlm --log smolvlm/gesture_outputs.log
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

2. **Saved Frames**  
   Snapshots stored in:
   - `opencv/frames_open cv/`
   - `smolvlm/frames_smolvlm/`

3. **Benchmarks Summary**  
   - `benchmarks/results_summary.csv`

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
