import cv2
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

JSON_PATH = "output/vehicle_plate_detections.json"

OUTPUT_DIR = "output/quality_plates"

TOP_N = None

MIN_WIDTH = 30
MIN_HEIGHT = 8


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DETECTIONS
# ============================================================

with open(JSON_PATH, "r") as f:
    detections = json.load(f)


print("\n====================================")
print("PLATE QUALITY ANALYZER")
print("====================================")

print("Total plate detections:", len(detections))


# ============================================================
# CALCULATE QUALITY
# ============================================================

quality_results = []


for detection in detections:

    crop_path = detection.get("plate_crop")

    if not crop_path:
        continue

    image = cv2.imread(crop_path)

    if image is None:
        continue


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]

    area = width * height


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = gray.mean()


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = gray.std()


    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    confidence = float(
        detection.get(
            "plate_confidence",
            0
        )
    )


    # --------------------------------------------------------
    # QUALITY SCORE
    #
    # We use:
    # - image size
    # - sharpness
    # - contrast
    # - detector confidence
    # --------------------------------------------------------

    size_score = min(
        area / 1000,
        10
    )

    sharpness_score = min(
        sharpness / 100,
        10
    )

    contrast_score = min(
        contrast / 25,
        10
    )

    confidence_score = confidence * 10


    quality_score = (
        size_score * 0.30
        +
        sharpness_score * 0.35
        +
        contrast_score * 0.15
        +
        confidence_score * 0.20
    )


    quality_results.append({

        "frame_number":
            detection.get(
                "frame_number"
            ),

        "plate_crop":
            crop_path,

        "width":
            width,

        "height":
            height,

        "area":
            area,

        "sharpness":
            round(
                sharpness,
                2
            ),

        "brightness":
            round(
                brightness,
                2
            ),

        "contrast":
            round(
                contrast,
                2
            ),

        "confidence":
            round(
                confidence,
                3
            ),

        "quality_score":
            round(
                quality_score,
                3
            )
    })


# ============================================================
# SORT
# ============================================================

quality_results.sort(
    key=lambda x: x["quality_score"],
    reverse=True
)


# ============================================================
# PRINT TOP RESULTS
# ============================================================

print("\n====================================")
print("TOP PLATE CROPS")
print("====================================")

for i, result in enumerate(
    quality_results,
    start=1
):

    print(
        f"{i:02d}. "
        f"Frame={result['frame_number']} | "
        f"Size={result['width']}x{result['height']} | "
        f"Sharpness={result['sharpness']:.1f} | "
        f"Confidence={result['confidence']:.2f} | "
        f"Score={result['quality_score']:.2f}"
    )


# ============================================================
# SAVE TOP PLATE IMAGES
# ============================================================

for i, result in enumerate(
    quality_results,
    start=1
):

    source = result["plate_crop"]

    image = cv2.imread(source)

    if image is None:
        continue


    output_path = os.path.join(
        OUTPUT_DIR,
        f"best_{i:02d}.jpg"
    )


    cv2.imwrite(
        output_path,
        image
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    "output/plate_quality_report.json"
)


with open(
    report_path,
    "w"
) as f:

    json.dump(
        quality_results,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("QUALITY ANALYSIS COMPLETED")
print("====================================")

print(
    "Analyzed:",
    len(quality_results)
)

print(
    "Image saved:",
    len(quality_results)
)

print("\nBest images:")
print(OUTPUT_DIR)

print("\nReport:")
print(report_path)