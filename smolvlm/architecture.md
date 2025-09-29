# System Architecture Diagram

```mermaid
graph TD

    %% Hardware
    subgraph Hardware
        Camera["📷 Camera
        Input: People showing hands/gestures
        Output: Live video frames"]

        Computer["💻 Computer
        Input: Video frames
        Output: Sends frames to software modules"]
    end

    %% Software
    subgraph Software
        OpenCV["🖼️ OpenCV / MediaPipe
        Input: Video frames
        Task: Detect hands and landmarks
        Output: Bounding boxes & landmarks"]

        HandRecognition["✋ Hand Recognition
        Input: Hand bounding boxes & landmarks
        Task: Identify hand side (Left/Right), finger count
        Output: Hand data"]

        GestureRecognition["👍/👎 Gesture Recognition
        Input: Hand data
        Task: Recognize gestures (Thumbs Up / Thumbs Down / Other)
        Output: Gesture type"]

        AttendanceManager["📒 Attendance Manager
        Input: Gesture type + person info
        Task: Mark attendance using gestures
        Output: Attendance record"]

        Logger["📝 Logger
        Input: System events & performance
        Task: Record errors, steps, CPU & memory usage
        Output: Log messages + performance details"]
    end

    %% Storage
    subgraph Storage
        GestureDB["🗂️ Gesture Database
        Stored: Reference gestures and mappings"]

        CSV["📂 Attendance.csv
        Stored: Name, ID, Date, Time, Status"]

        LogFile["📄 performance.log
        Stored: Events, Errors, CPU & Memory usage, Processing time"]
    end

    %% Connections
    Camera --> Computer
    Computer --> OpenCV
    OpenCV --> HandRecognition
    HandRecognition --> GestureRecognition
    GestureRecognition --> AttendanceManager
    HandRecognition -->|Compare with| GestureDB
    AttendanceManager --> CSV
    AttendanceManager --> Logger
    Logger --> LogFile
```
