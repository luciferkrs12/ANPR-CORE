from ultralytics import YOLO
import cv2
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/videos/traffic.mp4"

MODEL_PATH = "yolo11s.pt"

OUTPUT_DIR = "output"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "vehicle_detection_v2.mp4"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "vehicle_detections_v2.json"
)

# Initial confidence threshold
CONFIDENCE_THRESHOLD = 0.30


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("\n====================================")
print("VIDEO INFORMATION")
print("====================================")

print("FPS          :", fps)
print("Width        :", width)
print("Height       :", height)
print("Total Frames :", total_frames)

print("====================================")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# DETECTION STORAGE
# ============================================================

all_detections = []

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Run YOLO
    results = model(
        frame,
        verbose=False
    )

    frame_detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # --------------------------------------------
            # CLASS
            # --------------------------------------------

            class_id = int(
                box.cls[0]
            )

            # Ignore non-vehicle objects
            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_type = VEHICLE_CLASSES[
                class_id
            ]

            # --------------------------------------------
            # CONFIDENCE
            # --------------------------------------------

            confidence = float(
                box.conf[0]
            )

            # Ignore weak detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # --------------------------------------------
            # BOUNDING BOX
            # --------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # --------------------------------------------
            # SAVE DETECTION
            # --------------------------------------------

            detection = {

                "frame_number": frame_number,

                "vehicle_type": vehicle_type,

                "confidence": round(
                    confidence,
                    4
                ),

                "bounding_box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }

            frame_detections.append(
                detection
            )

            # --------------------------------------------
            # DRAW BOX
            # --------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # --------------------------------------------
            # LABEL
            # --------------------------------------------

            label = (
                f"{vehicle_type} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Store detections
    all_detections.extend(
        frame_detections
    )

    # Save processed frame
    video_writer.write(frame)

    # Progress
    if frame_number % 50 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

video_writer.release()


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_detections,
        file,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("VEHICLE DETECTION COMPLETED")
print("====================================")

print(
    "Frames processed:",
    frame_number
)

print(
    "Vehicle detections:",
    len(all_detections)
)

print(
    "Confidence threshold:",
    CONFIDENCE_THRESHOLD
)

print("\nOutput video:")
print(OUTPUT_VIDEO)

print("\nOutput JSON:")
print(OUTPUT_JSON)

print("====================================")