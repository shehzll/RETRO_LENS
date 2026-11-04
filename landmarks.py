
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load MediaPipe model
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Hand connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# Start webcam
cap = cv2.VideoCapture(0)

start_time = time.monotonic()
last_timestamp = -1

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp in milliseconds
    timestamp = int(
        (time.monotonic() - start_time) * 1000
    )

    timestamp = max(timestamp, last_timestamp + 1)
    last_timestamp = timestamp

    # Detect hands
    results = detector.detect_for_video(
        mp_image, timestamp
    )

    if results.hand_landmarks:

        for hand in results.hand_landmarks:

            points = []

            # Get landmark coordinates
            for lm in hand:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw hand connections
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 255),
                    2
                )

    cv2.imshow("Retro Lens - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
