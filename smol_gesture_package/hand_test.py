
import cv2
import mediapipe as mp

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Camera ---
cap = cv2.VideoCapture(0)  # Removed CAP_DSHOW for Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if results.multi_hand_landmarks:
        print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hands detected in this frame")

    cv2.imshow("Hand Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()