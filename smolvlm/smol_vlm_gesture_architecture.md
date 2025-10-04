# SmolVLM Real-Time Gesture Detection — Architecture

> **Purpose:** Architecture diagram and supporting notes for a real-time gesture detection system that uses SmolVLM for multimodal (image+text) inference. This document contains a diagram (Mermaid), component descriptions, data flow, deployment considerations, and suggestions for performance and reliability.

---

## 1. High-level Overview

The system captures live camera frames, preprocesses them, optionally extracts hand/pose keypoints, and sends one or more images (or image tokens) along with a prompt to SmolVLM for multimodal inference. Results are post-processed into gesture labels and actions, logged, and presented to the user or other systems.

---

## 2. Architecture Diagram (Mermaid)

```mermaid
flowchart TD
  Camera[Camera / Video Capture]
  Preproc[Preprocessing
  (resize, normalize,
  color conversion)]
  FrameBuf[Frame Buffer / Queue]
  Keypoint[Optional: Hand/Pose
  Keypoint Extraction]
  ImageTokens[Image Encoding
  & Tokenization]
  PromptPrep[Prompt Builder
  (text + <image> tokens)]
  SmolVLM[SmolVLM Inference
  (multimodal model)]
  Postproc[Post-processing
  (confidence, smoothing)]
  GestureMap[Gesture Mapping
  & Decision Logic]
  UI[UI / Overlay / API]
  Logger[Logging & Metrics]
  Storage[Storage
  (frames, logs)]
  Scheduler[Worker Pool / Scheduler]
  ModelCache[Model Loader
  & Cache]
  HW[Hardware: CPU/GPU/NPU]

  Camera --> Preproc --> FrameBuf
  FrameBuf -->|worker threads| Keypoint
  FrameBuf -->|worker threads| ImageTokens
  Keypoint --> ImageTokens
  ImageTokens --> PromptPrep
  PromptPrep --> ModelCache --> SmolVLM
  SmolVLM --> Postproc --> GestureMap --> UI
  GestureMap --> Logger
  UI --> Storage
  Logger --> Storage
  Scheduler --> FrameBuf
  HW --> ModelCache
  HW --> SmolVLM
```

---

## 3. Components — Detailed

**Camera / Video Capture**
- Captures frames at target FPS (e.g., 15–30 FPS).
- Should support hardware acceleration if available.

**Preprocessing**
- Resize, convert color space (BGR->RGB), normalize, and crop.
- Optionally do ROI selection (hand region) to reduce input size.

**Frame Buffer / Queue**
- Bounded queue to decouple capture from inference.
- Helps avoid blocking the camera if inference lags.

**Keypoint Extraction (Optional)**
- Lightweight hand/pose estimator (e.g., MediaPipe) to extract hand keypoints or ROI.
- Use to decide whether to send a frame to SmolVLM (saves computation).

**Image Encoding & Tokenization**
- Convert image(s) to the format SmolVLM expects (e.g., image tokens / embedding pipeline).
- If SmolVLM expects a fixed number of image tokens, ensure prompt + images align.

**Prompt Builder**
- Compose text prompt and insert `<image>` tokens where needed.
- Include context like prior gesture history, time, user ID, or task-specific instructions.

**Model Loader & Cache**
- Load SmolVLM once at startup; keep in memory.
- Keep batch/pipeline ready for inference.

**SmolVLM Inference**
- Runs the multimodal model on combined prompt + image tokens.
- Prefer GPU or NPU for low-latency.

**Post-processing & Smoothing**
- Convert SmolVLM output to discrete gesture labels.
- Apply temporal smoothing (e.g., majority window, exponential filter) to reduce flicker.

**Gesture Mapping & Decision Logic**
- Map textual model outputs to app actions (e.g., swipe-left -> next slide).
- Apply confidence and debounce thresholds.

**UI / Overlay / API**
- Display live overlay, recognized gestures, and confidence.
- Expose REST/WebSocket APIs for external integration.

**Logging & Storage**
- Capture inference latency, CPU/GPU usage, and detected gestures.
- Optionally save frames for positive/negative examples to a dataset folder.

**Worker Pool / Scheduler**
- Manage concurrent tasks: capture, inference, I/O.
- Use separate threads/processes to avoid GIL contention (Python) — consider multiprocessing.

**Hardware**
- Target environments: Desktop with GPU, Mobile with NPU, Edge devices with VPU.
- Fallback mode for CPU-only with reduced frame rate / resolution.

---

## 4. Data Flow (Sequence)

1. Camera captures frame -> enqueue to Frame Buffer.
2. Preprocessing worker dequeues frame -> resize & normalize.
3. Optional keypoint worker analyzes frame; if no hands detected, skip inference.
4. Image tokens are prepared and Prompt Builder composes multimodal prompt.
5. Scheduler assigns the prompt to a SmolVLM worker (ModelCache ensures model is loaded).
6. SmolVLM returns a textual/prediction output.
7. Post-processing converts output into gesture + confidence; smoothing is applied.
8. Gesture Mapping triggers UI update, logs event, and optionally stores the frame.

---

## 5. Performance & Reliability Tips

- **Batching:** If latency budget allows, batch multiple frames or requests to improve throughput. Beware of increased latency per request.
- **Quantization / Pruning:** Use INT8/FP16 quantized model variants to reduce memory and latency.
- **Model Warm-up:** Run a few dummy inferences at startup to cache kernels and avoid initial latency spike.
- **Asynchronous IO:** Use async I/O for logging and storage to avoid blocking inference.
- **Backpressure:** Drop older frames when the queue is full to prioritize fresher frames.
- **Profiling:** Measure time for capture, preprocessing, encoding, inference, postprocessing separately.
- **Fallback Layers:** If SmolVLM fails or is slow, have a lightweight fallback classifier for essential gestures.

---

## 6. Deployment / Scalability Patterns

- **Single-device, low-latency:** Keep everything local — capture, inference, UI.
- **Edge + Cloud hybrid:** Run lightweight detection on device; upload key frames to cloud SmolVLM for richer interpretation.
- **Server-side inference cluster:** For many clients, run SmolVLM on GPU servers and stream results back via websocket.

---

## 7. ML-specific Notes (SmolVLM-specific)

- Ensure the number of `<image>` tokens in the text prompt equals the number of images passed — mismatches cause errors.
- Wrap images and prompt text consistently; maintain same token ordering expected by the model.
- Cache embeddings where possible if you repeatedly send similar frames or repeated prompts.

---

## 8. Example File / Folder Layout

```
project/
├─ app/
│  ├─ capture.py
│  ├─ preprocess.py
│  ├─ keypoint.py
│  ├─ inference.py
│  ├─ prompt_builder.py
│  └─ ui.py
├─ models/
│  └─ smolvlm/
├─ logs/
├─ saved_frames/
└─ requirements.txt
```

---

## 9. Quick Checklist Before Implementation

- [ ] Decide whether to run keypoint detector locally to filter frames.
- [ ] Confirm SmolVLM input format (image token count, prompt formatting).
- [ ] Choose target hardware and quantization strategy.
- [ ] Implement bounded queues and a worker pool.
- [ ] Add profiling and logging.

---

If you want, I can:
- convert the Mermaid diagram into a PNG/SVG,
- produce a PlantUML diagram,
- or embed a simpler ASCII diagram for terminals.

Tell me which output you prefer and any details to include (e.g., exact model names, target FPS, hardware constraints).

