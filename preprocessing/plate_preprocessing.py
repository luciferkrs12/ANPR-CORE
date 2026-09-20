import cv2
import os
import glob


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "output/plates"
OUTPUT_DIR = "output/processed_plates"

UPSCALE_FACTOR = 4


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PREPROCESS PLATE
# ============================================================

def preprocess_plate(image):

    # --------------------------------------------------------
    # STEP 1 — UPSCALE
    # --------------------------------------------------------

    height, width = image.shape[:2]

    new_width = width * UPSCALE_FACTOR
    new_height = height * UPSCALE_FACTOR

    upscaled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )


    # --------------------------------------------------------
    # STEP 2 — GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # STEP 3 — CONTRAST ENHANCEMENT
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)


    # --------------------------------------------------------
    # STEP 4 — DENOISING
    # --------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )


    # --------------------------------------------------------
    # STEP 5 — SHARPENING
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        denoised,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blurred,
        -0.5,
        0
    )


    # --------------------------------------------------------
    # STEP 6 — ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    thresholded = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )


    return {
        "upscaled": upscaled,
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "sharpened": sharpened,
        "thresholded": thresholded
    }


# ============================================================
# FIND PLATE IMAGES
# ============================================================

image_paths = glob.glob(
    os.path.join(
        INPUT_DIR,
        "*.jpg"
    )
)


print("\n====================================")
print("LICENSE PLATE PREPROCESSING")
print("====================================")

print(
    "Input plate images:",
    len(image_paths)
)


# ============================================================
# PROCESS
# ============================================================

processed_count = 0


for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not read:",
            image_path
        )

        continue


    results = preprocess_plate(image)


    filename = os.path.basename(
        image_path
    )

    name = os.path.splitext(
        filename
    )[0]


    # --------------------------------------------------------
    # SAVE ENHANCED IMAGE
    # --------------------------------------------------------

    enhanced_path = os.path.join(
        OUTPUT_DIR,
        name + "_enhanced.jpg"
    )

    cv2.imwrite(
        enhanced_path,
        results["sharpened"]
    )


    # --------------------------------------------------------
    # SAVE THRESHOLD IMAGE
    # --------------------------------------------------------

    threshold_path = os.path.join(
        OUTPUT_DIR,
        name + "_threshold.jpg"
    )

    cv2.imwrite(
        threshold_path,
        results["thresholded"]
    )


    processed_count += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("PREPROCESSING COMPLETED")
print("====================================")

print(
    "Images processed:",
    processed_count
)

print(
    "Upscale factor:",
    UPSCALE_FACTOR
)

print("\nOutput directory:")
print(OUTPUT_DIR)