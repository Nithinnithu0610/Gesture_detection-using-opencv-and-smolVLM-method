# Data Flow Diagram (DFD)

```mermaid
flowchart TD
    Camera -->|Captures Frame| FaceRecognition[Face Recognition Module]
    Camera -->|Captures Frame| GestureRecognition[Gesture Recognition Module]

    FaceRecognition -->|Encodes Face| Encoder[Face Encoding]
    Encoder -->|Compare with Stored Encodings| AttendanceCheck[Attendance Manager]
    AttendanceCheck -->|If new entry| CSV[(Attendance.csv)]
    AttendanceCheck -->|Logs performance| LogFile[(performance.log)]

    GestureRecognition --> HandSide[Hand Side Detection]
    GestureRecognition --> FingerCount[Finger Counting]
    GestureRecognition --> GestureType[Gesture Recognition]

    HandSide --> GestureLogger[Logger]
    FingerCount --> GestureLogger
    GestureType --> GestureLogger
    GestureLogger --> GestureLogFile[(gesture.log)]

    CSV --> Teacher[Teacher/Admin View]
    LogFile --> Developer[Developer]
    GestureLogFile --> Developer[Developer]
```
