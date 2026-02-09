import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# Initialize MediaPipe Hand Tracking
mp_hands = mp.tasks.vision.HandLandmarker
model_path = 'hand_landmarker.task'

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,  # Changed parameter name
    min_hand_presence_confidence=0.7,    # Added this parameter
    min_tracking_confidence=0.7          # Added this parameter
)

# 3. Create the HandLandmarker task instance
hands = vision.HandLandmarker.create_from_options(options)
mp_draw = mp.tasks.vision.drawing_utils


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

points = [[] for _ in range(4)]

paintWindow = np.zeros((480, 640, 3)) + 255

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_height, h_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = hands.detect(mp_image)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (160 + i * 115, 1), (255 + i * 115, 65), color, -1)

    cv2.putText(frame, "CLEAR", (60, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            index_finger = hand_landmarks[8]
            cx, cy = int(index_finger.x * h_width), int(index_finger.y * h_height)

            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)

            if cy <= 65:
                if 40 <= cx <= 140:
                    points = [[] for _ in range(4)]
                    paintWindow[67:, :, :] = 255
                elif 160 <= cx <= 255:
                    colorIndex = 0
                elif 275 <= cx <= 370:
                    colorIndex = 1
                elif 390 <= cx <= 485:
                    colorIndex = 2
                elif 505 <= cx <= 600:
                    colorIndex = 3
            else:
                if not points[colorIndex] or len(points[colorIndex][-1]) == 0:
                    points[colorIndex].append(deque(maxlen=512))
                points[colorIndex][-1].appendleft((cx, cy))
    else:
        for i in range(4):
            if points[i] and len(points[i][-1]) > 0:
                points[i].append(deque(maxlen=512))

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()