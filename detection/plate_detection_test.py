from ultralytics import YOLO
import cv2
import os


# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "models/plate_best.pt"
VIDEO_PATH = "data/videos/traffic.mp4"

OUTPUT_PATH = "output/plate_detection_test.mp4"

CONFIDENCE_THRESHOLD = 0.25


# ============================================
# LOAD MODEL
# ============================================

print("\n====================================")
print("LOADING LICENSE PLATE MODEL")
print("====================================")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# ============================================
# OPEN VIDEO
# ============================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("\n====================================")
print("VIDEO INFORMATION")
print("====================================")
print("FPS          :", fps)
print("Width        :", width)
print("Height       :", height)
print("Total Frames :", total_frames)


# ============================================
# OUTPUT VIDEO
# ============================================

os.makedirs("output", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================
# PROCESS VIDEO
# ============================================

frame_number = 0
plate_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            plate_count += 1

            # Draw plate box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"PLATE {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    out.write(frame)

    if frame_number % 50 == 0:
        print(
            f"Processed {frame_number}/{total_frames}"
        )


# ============================================
# RELEASE
# ============================================

cap.release()
out.release()


print("\n====================================")
print("PLATE DETECTION TEST COMPLETED")
print("====================================")

print("Frames processed :", frame_number)
print("Plate detections  :", plate_count)
print("Confidence        :", CONFIDENCE_THRESHOLD)

print("\nOutput:")
print(OUTPUT_PATH)