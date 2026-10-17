import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

st.set_page_config(page_title="Smart Wound Monitor", page_icon="🩹", layout="centered")

# These are placeholders from the previous app, not real laboratory calibration data.
# Replace with measured values from known pH buffer solutions before enabling estimates.
CALIBRATION_POINTS = [
    {"ph": 5.5, "rgb": [100, 200, 100]},
    {"ph": 6.5, "rgb": [200, 200, 100]},
    {"ph": 7.5, "rgb": [250, 150, 50]},
    {"ph": 8.5, "rgb": [200, 50, 50]},
]
CALIBRATION_IS_VALIDATED = False

def detect_indicator(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([145, 25, 25]), np.array([179, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([125, 25, 25]), np.array([144, 255, 255]))
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    height, width = mask.shape
    y1, y2 = int(height * 0.20), int(height * 0.90)
    x1, x2 = int(width * 0.15), int(width * 0.85)
    roi = cv2.medianBlur(mask[y1:y2, x1:x2], 5)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(80, height * width * 0.0002):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h) if h else 0
        score = area * (1 - min(abs(1 - aspect), 1)) * (0.5 + circularity)
        candidates.append((score, x, y, w, h))
    if not candidates:
        return None
    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    return x + x1, y + y1, w, h

def estimate_ph(target_rgb):
    target = np.array(target_rgb, dtype=float)
    distances = []
    for point in CALIBRATION_POINTS:
        d = float(np.linalg.norm(target - np.array(point["rgb"], dtype=float)))
        distances.append((d, point["ph"]))
    distances.sort(key=lambda item: item[0])
    (d1, ph1), (d2, ph2) = distances[:2]
    if d1 + d2 == 0:
        return ph1
    return round((ph1 * d2 + ph2 * d1) / (d1 + d2), 2)

camera_image = st.camera_input("Capture Dressing Patch")
with st.expander("Or upload a photo instead"):
    upload_image = st.file_uploader("Choose a patch photo", type=["png", "jpg", "jpeg"])
image_file = camera_image if camera_image is not None else upload_image

if image_file is None:
    st.info("Capture a clear photo of the dressing patch to begin.")
else:
    try:
        pil_image = Image.open(image_file).convert("RGB")
        rgb = np.array(pil_image)
        box = detect_indicator(rgb)

        st.subheader("Captured photo")
        if box is None:
            st.image(pil_image, use_container_width=True)
            st.warning("Indicator circle not detected. Try again with clear lighting and the full circle visible.")
        else:
            x, y, w, h = box
            outlined = pil_image.copy()
            draw = ImageDraw.Draw(outlined)
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=max(3, rgb.shape[1] // 250))
            st.image(outlined, caption="Detected indicator area", use_container_width=True)

            padx, pady = max(1, int(w * 0.2)), max(1, int(h * 0.2))
            sx1, sx2 = max(0, x + padx), min(rgb.shape[1], x + w - padx)
            sy1, sy2 = max(0, y + pady), min(rgb.shape[0], y + h - pady)
            sample = rgb[sy1:sy2, sx1:sx2]
            if sample.size == 0:
                sample = rgb[y:y+h, x:x+w]
            measured_rgb = np.median(sample.reshape(-1, 3), axis=0).astype(int).tolist()

            st.divider()
            st.subheader("AI Analysis Results")

            if CALIBRATION_IS_VALIDATED:
                ph = estimate_ph(measured_rgb)
                st.metric("Calculated Indicator pH", f"pH {ph}")
                if ph <= 6.0:
                    st.success("Acidic indicator response (pH ≤ 6.0). This does not establish that the wound is healthy.")
                elif ph <= 7.0:
                    st.info("Indicator response in the pH 6.0–7.0 range. This does not establish wound healing.")
                elif ph <= 8.0:
                    st.warning("Indicator response in the pH 7.0–8.0 range. Clinical condition cannot be determined from pH alone.")
                else:
                    st.error("Indicator response above pH 8.0. This is not proof of infection; seek clinical assessment if concerned.")
            else:
                st.metric("Calculated Indicator pH", "Calibration required")
                st.info(
                    "The previous app's RGB-to-pH values were placeholders, not measured calibration data. "
                    "Add validated measurements from known pH buffer solutions to enable a pH estimate."
                )

            st.subheader("Wound Condition")
            st.warning(
                "Cannot be determined from patch colour alone. This prototype cannot confirm healthy tissue, healing progress, or infection."
            )
            st.caption("Use consistent lighting and validate the indicator with known pH buffers before relying on any pH estimate.")
    except Exception as error:
        st.error("The photo could not be processed. Please try another clear JPG or PNG photo.")
        st.caption(f"Technical detail: {error}")
