from ultralytics import YOLO
import cv2
import json
import os
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PLATE_MODEL_PATH = "models/plate_best.pt"

VIDEO_PATH = "data/videos/traffic.mp4"

VEHICLE_JSON_PATH = "output/vehicle_detections_v2.json"

OUTPUT_VIDEO_PATH = "output/vehicle_plate_detection.mp4"

OUTPUT_JSON_PATH = "output/vehicle_plate_detections.json"

PLATE_CROP_DIR = "output/plates"

PLATE_CONFIDENCE_THRESHOLD = 0.25

# Add a little padding around the vehicle crop.
# This helps when the vehicle bounding box is slightly tight.
VEHICLE_PADDING = 10


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("output", exist_ok=True)
os.makedirs(PLATE_CROP_DIR, exist_ok=True)


# ============================================================
# LOAD PLATE MODEL
# ============================================================

print("\n====================================")
print("LOADING LICENSE PLATE MODEL")
print("====================================")

plate_model = YOLO(PLATE_MODEL_PATH)

print("Plate model loaded successfully!")


# ============================================================
# LOAD VEHICLE DETECTIONS
# ============================================================

print("\n====================================")
print("LOADING VEHICLE DETECTIONS")
print("====================================")

with open(VEHICLE_JSON_PATH, "r") as f:
    vehicle_detections = json.load(f)

print("Vehicle detections loaded:", len(vehicle_detections))


# ============================================================
# GROUP VEHICLES BY FRAME
# ============================================================

vehicles_by_frame = defaultdict(list)

for detection in vehicle_detections:

    frame_number = detection.get("frame_number")

    if frame_number is None:
        continue

    vehicles_by_frame[frame_number].append(detection)


print(
    "Frames containing vehicle detections:",
    len(vehicles_by_frame)
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


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


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

vehicle_count = 0

plate_count = 0

saved_crop_count = 0

results_json = []


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Get vehicles detected in this frame
    # --------------------------------------------------------

    current_vehicles = vehicles_by_frame.get(
        frame_number,
        []
    )

    vehicle_index = 0

    # --------------------------------------------------------
    # Process every vehicle
    # --------------------------------------------------------

    for vehicle in current_vehicles:

        vehicle_index += 1

        vehicle_count += 1

        vehicle_type = vehicle.get(
            "vehicle_type",
            "unknown"
        )

        vehicle_confidence = float(
            vehicle.get(
                "confidence",
                0.0
            )
        )

        bbox = vehicle.get(
            "bounding_box"
        )

        if not bbox or len(bbox) != 4:
            continue

        # ----------------------------------------------------
        # Our vehicle JSON uses:
        #
        # [x1, y1, x2, y2]
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        # ----------------------------------------------------
        # Add padding
        # ----------------------------------------------------

        x1_crop = max(
            0,
            x1 - VEHICLE_PADDING
        )

        y1_crop = max(
            0,
            y1 - VEHICLE_PADDING
        )

        x2_crop = min(
            width,
            x2 + VEHICLE_PADDING
        )

        y2_crop = min(
            height,
            y2 + VEHICLE_PADDING
        )

        # ----------------------------------------------------
        # Validate crop
        # ----------------------------------------------------

        if x2_crop <= x1_crop:
            continue

        if y2_crop <= y1_crop:
            continue

        vehicle_crop = frame[
            y1_crop:y2_crop,
            x1_crop:x2_crop
        ]

        if vehicle_crop.size == 0:
            continue

        # ----------------------------------------------------
        # Run plate detector INSIDE vehicle crop
        # ----------------------------------------------------

        plate_results = plate_model(
            vehicle_crop,
            conf=PLATE_CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # ----------------------------------------------------
        # Process plate detections
        # ----------------------------------------------------

        for result in plate_results:

            if result.boxes is None:
                continue

            for plate_box in result.boxes:

                plate_confidence = float(
                    plate_box.conf[0]
                )

                # Coordinates relative to vehicle crop
                px1, py1, px2, py2 = map(
                    int,
                    plate_box.xyxy[0].tolist()
                )

                # ------------------------------------------------
                # Convert vehicle-crop coordinates
                # back to FULL FRAME coordinates
                # ------------------------------------------------

                full_x1 = px1 + x1_crop
                full_y1 = py1 + y1_crop

                full_x2 = px2 + x1_crop
                full_y2 = py2 + y1_crop

                # ------------------------------------------------
                # Clamp coordinates to frame
                # ------------------------------------------------

                full_x1 = max(
                    0,
                    min(width - 1, full_x1)
                )

                full_y1 = max(
                    0,
                    min(height - 1, full_y1)
                )

                full_x2 = max(
                    0,
                    min(width, full_x2)
                )

                full_y2 = max(
                    0,
                    min(height, full_y2)
                )

                if full_x2 <= full_x1:
                    continue

                if full_y2 <= full_y1:
                    continue

                # ------------------------------------------------
                # Extract actual plate crop
                # ------------------------------------------------

                plate_crop = frame[
                    full_y1:full_y2,
                    full_x1:full_x2
                ]

                if plate_crop.size == 0:
                    continue

                # ------------------------------------------------
                # Save plate image
                # ------------------------------------------------

                saved_crop_count += 1

                crop_filename = (
                    f"plate_{saved_crop_count:06d}.jpg"
                )

                crop_path = os.path.join(
                    PLATE_CROP_DIR,
                    crop_filename
                )

                cv2.imwrite(
                    crop_path,
                    plate_crop
                )

                # ------------------------------------------------
                # Draw vehicle box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                vehicle_label = (
                    f"{vehicle_type} "
                    f"{vehicle_confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    vehicle_label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    2
                )

                # ------------------------------------------------
                # Draw plate box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (full_x1, full_y1),
                    (full_x2, full_y2),
                    (0, 255, 0),
                    2
                )

                plate_label = (
                    f"PLATE "
                    f"{plate_confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    plate_label,
                    (
                        full_x1,
                        max(full_y1 - 8, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # Save JSON record
                # ------------------------------------------------

                plate_count += 1

                timestamp_seconds = (
                    (frame_number - 1) / fps
                    if fps > 0
                    else 0
                )

                timestamp_ms = int(
                    timestamp_seconds * 1000
                )

                record = {
                    "frame_number": frame_number,

                    "timestamp_seconds":
                        round(
                            timestamp_seconds,
                            3
                        ),

                    "timestamp_ms":
                        timestamp_ms,

                    "vehicle_index":
                        vehicle_index,

                    "vehicle_type":
                        vehicle_type,

                    "vehicle_confidence":
                        round(
                            vehicle_confidence,
                            4
                        ),

                    "vehicle_bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "plate_confidence":
                        round(
                            plate_confidence,
                            4
                        ),

                    "plate_bbox": [
                        full_x1,
                        full_y1,
                        full_x2,
                        full_y2
                    ],

                    "plate_crop":
                        crop_path
                }

                results_json.append(record)

    # --------------------------------------------------------
    # Write processed frame
    # --------------------------------------------------------

    out.write(frame)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()
out.release()


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON_PATH,
    "w"
) as f:

    json.dump(
        results_json,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("VEHICLE → PLATE DETECTION COMPLETED")
print("====================================")

print(
    "Frames processed      :",
    frame_number
)

print(
    "Vehicle detections    :",
    vehicle_count
)

print(
    "Plate detections      :",
    plate_count
)

print(
    "Plate crops saved     :",
    saved_crop_count
)

print(
    "Plate confidence      :",
    PLATE_CONFIDENCE_THRESHOLD
)

print("\nOutput video:")
print(OUTPUT_VIDEO_PATH)

print("\nOutput JSON:")
print(OUTPUT_JSON_PATH)

print("\nPlate crops:")
print(PLATE_CROP_DIR)