
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Smart Wound Monitor",
    page_icon="🩹",
    layout="centered"
)

st.title("🩹 AI-Assisted Smart Wound Monitor")
st.write("Capture a clear photo of the herbal dressing indicator.")

# IMPORTANT:
# Replace these placeholder values with actual laboratory measurements.
# Keep False until you have validated your calibration data.
CALIBRATION_IS_VALIDATED = False

CALIBRATION_POINTS = [
    {"ph": 5.5, "rgb": [100, 200, 100]},
    {"ph": 6.5, "rgb": [200, 200, 100]},
    {"ph": 7.5, "rgb": [250, 150, 50]},
    {"ph": 8.5, "rgb": [200, 50, 50]},
]


def detect_indicator(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(
        hsv, np.array([0, 35, 35]),
        np.array([12, 255, 255])
    )
    mask2 = cv2.inRange(
        hsv, np.array([145, 25, 25]),
        np.array([179, 255, 255])
    )
    mask3 = cv2.inRange(
        hsv, np.array([125, 25, 25]),
        np.array([144, 255, 255])
    )

    mask = cv2.bitwise_or(
        mask1, cv2.bitwise_or(mask2, mask3)
    )

    height, width = mask.shape

    y1 = int(height * 0.20)
    y2 = int(height * 0.90)
    x1 = int(width * 0.15)
    x2 = int(width * 0.85)

    roi = cv2.medianBlur(mask[y1:y2, x1:x2], 5)

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < max(80, height * width * 0.0002):
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4 * np.pi * area / (perimeter * perimeter)
        )

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h) if h else 0

        score = (
            area
            * (1 - min(abs(1 - aspect), 1))
            * (0.5 + circularity)
        )

        candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(
        candidates, key=lambda item: item[0]
    )

    return x + x1, y + y1, w, h


def calculate_ph(target_rgb):
    target = np.array(target_rgb, dtype=float)
    distances = []

    for point in CALIBRATION_POINTS:
        calibration_rgb = np.array(
            point["rgb"], dtype=float
        )
        distance = np.linalg.norm(target - calibration_rgb)
        distances.append((distance, point["ph"]))

    distances.sort(key=lambda item: item[0])

    (d1, ph1), (d2, ph2) = distances[:2]

    if d1 + d2 == 0:
        return ph1

    result = (ph1 * d2 + ph2 * d1) / (d1 + d2)

    return round(result, 2)


camera_image = st.camera_input("Capture Dressing Patch")

with st.expander("Or upload a photo"):
    upload_image = st.file_uploader(
        "Choose photo",
        type=["png", "jpg", "jpeg"]
    )

image_file = (
    camera_image
    if camera_image is not None
    else upload_image
)

if image_file is None:
    st.info("Capture a clear photo to start analysis.")

else:
    try:
        image = Image.open(image_file).convert("RGB")
        image_array = np.array(image)

        box = detect_indicator(image_array)

        st.subheader("Captured Photo")

        if box is None:
            st.image(image, use_container_width=True)
            st.warning(
                "Indicator circle not detected. "
                "Try again with clear lighting and the "
                "full indicator circle visible."
            )

        else:
            x, y, w, h = box

            outlined_image = image.copy()
            draw = ImageDraw.Draw(outlined_image)

            draw.rectangle(
                [x, y, x + w, y + h],
                outline=(0, 255, 0),
                width=max(3, image_array.shape[1] // 250)
            )

            st.image(
                outlined_image,
                caption="Detected indicator area",
                use_container_width=True
            )

            # Measure colour from the centre of the indicator.
            padx = max(1, int(w * 0.20))
            pady = max(1, int(h * 0.20))

            sx1 = max(0, x + padx)
            sx2 = min(image_array.shape[1], x + w - padx)
            sy1 = max(0, y + pady)
            sy2 = min(image_array.shape[0], y + h - pady)

            sample = image_array[sy1:sy2, sx1:sx2]

            if sample.size == 0:
                sample = image_array[y:y + h, x:x + w]

            measured_rgb = np.median(
                sample.reshape(-1, 3), axis=0
            ).astype(int).tolist()

            st.divider()
            st.subheader("AI Analysis Results")

            if CALIBRATION_IS_VALIDATED:
                ph = calculate_ph(measured_rgb)

                st.metric(
                    "Calculated Indicator pH",
                    f"pH {ph}"
                )

                st.subheader("Wound Condition")

                st.warning(
                    "The pH result alone cannot establish "
                    "whether a wound is healthy, healing, "
                    "or infected. Clinical assessment is needed."
                )

            else:
                st.metric(
                    "Calculated Indicator pH",
                    "Calibration required"
                )

                st.info(
                    "Enter and validate your actual hibiscus "
                    "indicator colour measurements using known "
                    "pH buffer solutions before enabling pH estimates."
                )

                st.subheader("Wound Condition")

                st.warning(
                    "Cannot be determined from the photo alone. "
                    "This prototype cannot confirm healing or infection."
                )

    except Exception as error:
        st.error(
            "The photo could not be processed. "
            "Please try another clear JPG or PNG photo."
        )
        st.caption(f"Technical detail: {error}")
