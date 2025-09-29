# Gesture Recognition Submission - Final (Unified Logs & Frames)

Created: 2025-09-29T12:34:28.935985

Structure:
- opencv/
  - gesture_opencv.py
  - gesture_outputs.log
  - frames/
  - architecture_diagram.pdf
  - dfd_diagram.pdf
- smolvlm/
  - gesture_smolvlm.py
  - gesture_outputs.log
  - frames/
  - architecture_diagram.pdf
  - dfd_diagram.pdf
- benchmarks/run_benchmarks.py
- requirements.txt
- README.md

Usage examples:
python opencv/gesture_opencv.py --out_dir opencv/frames --log opencv/gesture_outputs.log
python smolvlm/gesture_smolvlm.py --out_dir smolvlm/frames --log smolvlm/gesture_outputs.log
python benchmarks/run_benchmarks.py 20

Notes:
- Each module logs unified blocks (gesture + performance) to gesture_outputs.log.
- Saves up to 3 frames per (hand, gesture) combination into module's frames/ folder.
