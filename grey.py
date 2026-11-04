
import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# 1. LOAD HAND DETECTION MODEL
# --------------------------------

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------
# 2. START WEBCAM
# --------------------------------

cap = cv2.VideoCapture(0)

start_time = time.monotonic()
last_timestamp = -1

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(
        (time.monotonic() - start_time) * 1000
    )

    timestamp = max(timestamp, last_timestamp + 1)
    last_timestamp = timestamp

    results = detector.detect_for_video(
        mp_image, timestamp
    )


    # --------------------------------
    # 3. HAND LANDMARK DETECTION
    # --------------------------------

    if results.hand_landmarks:

        for hand in results.hand_landmarks:

            points = []

            for lm in hand:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            # Draw landmarks
            for x, y in points:

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 255),
                    2
                )


            # --------------------------------
            # 4. CREATE POLYGON
            # --------------------------------

            # Thumb, index, middle and pinky tips
            selected_points = np.array([
                points[4],
                points[8],
                points[12],
                points[20]
            ], dtype=np.int32)

            # Draw polygon boundary
            cv2.polylines(
                frame,
                [selected_points],
                True,
                (0, 255, 255),
                2
            )


            # --------------------------------
            # 5. APPLY GREYSCALE FILTER
            # --------------------------------

            # Convert complete frame to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

            # Create black mask
            mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            # Fill polygon with white
            cv2.fillPoly(
                mask,
                [selected_points],
                255
            )

            # Apply grayscale only inside polygon
            frame[mask == 255] = gray[mask == 255]


    # --------------------------------
    # 6. DISPLAY OUTPUT
    # --------------------------------

    cv2.imshow(
        "Retro Lens - Grayscale Filter",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
detector.close()
cv2.destroyAllWindows()
