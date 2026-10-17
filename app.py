import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

st.set_page_config(page_title="Smart Herbal Wound Monitor", page_icon="🩹", layout="centered")

# IMPORTANT:
# These are example values copied from the old app and were marked as placeholders.
# Replace them with real measurements from known pH buffer solutions before using pH estimates.
CALIBRATION_POINTS = [
    {"ph": 5.5, "rgb": [100, 200, 100]},
    {"ph": 6.5, "rgb": [200, 200, 100]},
    {"ph": 7.5, "rgb": [250, 150, 50]},
    {"ph": 8.5, "rgb": [200, 50, 50]},
]
CALIBRATION_IS_VALIDATED = False

def detect_indicator(rgb_image):
    """Find a likely pink/purple indicator region near the centre/lower-centre."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([145, 25, 25]), np.array([179, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([125, 25, 25]), np.array([144, 255, 255]))
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    height, width = mask.shape
    roi_y1, roi_y2 = int(height * 0.20), int(height * 0.90)
    roi_x1, roi_x2 = int(width * 0.15), int(width * 0.85)
    roi = cv2.medianBlur(mask[roi_y1:roi_y2, roi_x1:roi_x2], 5)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(80, (height * width) * 0.0002):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h) if h else 0
        score = area * (1.0 - min(abs(1.0 - aspect), 1.0)) * (0.5 + circularity)
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None
    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    return (x + roi_x1, y + roi_y1, w, h)

def estimate_ph_from_rgb(target_rgb):
    """Nearest-colour interpolation; only meaningful with experimentally validated calibration."""
    target = np.array(target_rgb, dtype=float)
    distances = []
    for point in CALIBRATION_POINTS:
        cal_rgb = np.array(point["rgb"], dtype=float)
        distances.append((float(np.linalg.norm(target - cal_rgb)), point["ph"]))
    distances.sort(key=lambda item: item[0])
    (d1, ph1), (d2, ph2) = distances[:2]
    if d1 + d2 == 0:
        return float(ph1)
    return round((ph1 * d2 + ph2 * d1) / (d1 + d2), 2)

camera_image = st.camera_input("Capture Dressing Patch")
with st.expander("Or upload a photo instead"):
    uploaded_image = st.file_uploader("Choose a patch photo", type=["png", "jpg", "jpeg"])
image_file = camera_image if camera_image is not None else uploaded_image

if image_file is None:
    st.info("Capture a clear photo of the patch to begin.")
else:
    try:
        pil_image = Image.open(image_file).convert("RGB")
        rgb_image = np.array(pil_image)

        detected_box = detect_indicator(rgb_image)
        st.subheader("Captured photo")

        if detected_box is None:
            st.image(pil_image, use_container_width=True)
            st.warning("I could not confidently find the pink/purple indicator circle. Try brighter, even lighting and keep the circle fully visible.")
        else:
            x, y, w, h = detected_box
            outlined = pil_image.copy()
            draw = ImageDraw.Draw(outlined)
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=max(3, rgb_image.shape[1] // 250))
            st.image(outlined, caption="Green outline indicates the region selected for colour measurement.", use_container_width=True)

            # Sample central portion of the detected region to reduce edge/background effects.
            pad_x, pad_y = max(1, int(w * 0.20)), max(1, int(h * 0.20))
            x1, x2 = max(0, x + pad_x), min(rgb_image.shape[1], x + w - pad_x)
            y1, y2 = max(0, y + pad_y), min(rgb_image.shape[0], y + h - pad_y)
            sample = rgb_image[y1:y2, x1:x2]
            if sample.size == 0:
                sample = rgb_image[y:y + h, x:x + w]
            avg_rgb = np.median(sample.reshape(-1, 3), axis=0).astype(int).tolist()
            r, g, b = avg_rgb
            colour_hex = f"#{r:02X}{g:02X}{b:02X}"
            avg_hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0].tolist()

            st.divider()
            st.subheader("AI Analysis Results")
            st.write("**Detected indicator colour**")
            st.markdown(
                f'<div style="background-color:{colour_hex};width:100%;height:44px;border-radius:8px;'
                f'border:1px solid #888;margin-bottom:10px;"></div>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("RGB", f"{r}, {g}, {b}")
            col2.metric("HSV", f"{avg_hsv[0]}, {avg_hsv[1]}, {avg_hsv[2]}")
            col3.metric("HEX", colour_hex)

            if CALIBRATION_IS_VALIDATED:
                estimated_ph = estimate_ph_from_rgb(avg_rgb)
                st.metric("Estimated indicator pH", f"pH {estimated_ph}")
                st.caption("This estimate is only valid within the range and conditions tested during calibration.")
            else:
                st.metric("Estimated pH", "Not calibrated")
                st.info(
                    "The previous app used placeholder RGB values, not measured laboratory calibration data. "
                    "Enter RGB measurements from known buffer solutions and validate the calibration before showing a pH estimate."
                )

            st.subheader("Wound condition")
            st.warning(
                "Not determined. Patch colour alone cannot confirm that a wound is healthy, healing, or infected. "
                "This prototype does not provide a clinical diagnosis."
            )
            st.caption(
                "For reliable pH estimation, photograph the indicator in consistent lighting alongside known pH buffer "
                "solutions, collect replicate RGB values, and validate the calibration on separate samples."
            )
    except Exception as error:
        st.error("The photo could not be processed. Please try another clear JPG or PNG photo.")
        st.caption(f"Technical detail: {error}")
