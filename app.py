import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

st.set_page_config(page_title="Herbal Wound Dressing Colour Analyzer", layout="centered")
st.title("AI-Assisted Smart Herbal Wound Dressing")
st.subheader("Automatic central indicator detection")

st.write(
    "Upload a clear photograph. The app will try to locate the dark pink-purple indicator circle, "
    "outline the selected area, and measure its colour."
)
st.warning(
    "The app measures photographed colour only. It cannot confirm neutral pH or wound health "
    "unless the indicator has been calibrated against known-pH solutions."
)

st.markdown("### Capture patch photograph")
camera_photo = st.camera_input("Take a photo of the central indicator")

st.markdown("Or upload a photo if the camera is unavailable.")
uploaded_photo = st.file_uploader("Upload patch photograph", type=["jpg", "jpeg", "png"])
photo = camera_photo if camera_photo is not None else uploaded_photo

def detect_indicator(rgb):
    """Find a likely pink/purple circular indicator using colour segmentation and shape."""
    h_img, w_img = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Search mainly in the lower-central region, where the small indicator sits
    # in the user's reference photograph. This can be adjusted if the photo framing changes.
    x_start, x_end = int(w_img * 0.20), int(w_img * 0.75)
    y_start, y_end = int(h_img * 0.45), int(h_img * 0.90)
    crop_hsv = hsv[y_start:y_end, x_start:x_end]

    hue = crop_hsv[:, :, 0]
    sat = crop_hsv[:, :, 1]
    val = crop_hsv[:, :, 2]

    # Pink/red/purple hues, with sufficient saturation to reduce clear dish/background pixels.
    hue_mask = ((hue <= 12) | ((hue >= 135) & (hue <= 179)))
    mask = (hue_mask & (sat >= 45) & (val >= 25) & (val <= 235)).astype(np.uint8) * 255

    # Join small gaps in the coloured circle and remove isolated specks.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    crop_area = crop_hsv.shape[0] * crop_hsv.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(100, crop_area * 0.001) or area > crop_area * 0.25:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 8:
            continue
        # Prefer reasonably round components near the centre of the search area.
        target_x, target_y = crop_hsv.shape[1] * 0.55, crop_hsv.shape[0] * 0.70
        distance = np.sqrt(((cx - target_x) / max(1, crop_hsv.shape[1]))**2 +
                           ((cy - target_y) / max(1, crop_hsv.shape[0]))**2)
        score = area * (0.35 + min(circularity, 1.0)) / (1 + 5 * distance)
        candidates.append((score, contour, circularity))

    if not candidates:
        return None

    _, contour, circularity = max(candidates, key=lambda item: item[0])
    x, y, w, h = cv2.boundingRect(contour)
    # Add a small margin around the detected coloured area.
    margin = max(3, int(max(w, h) * 0.08))
    x1 = max(0, x_start + x - margin)
    y1 = max(0, y_start + y - margin)
    x2 = min(w_img, x_start + x + w + margin)
    y2 = min(h_img, y_start + y + h + margin)
    return (x1, y1, x2, y2, circularity)

if photo is not None:
    image = Image.open(photo).convert("RGB")
    rgb = np.array(image)
    result = detect_indicator(rgb)

    if result is None:
        st.image(image, caption="Uploaded photograph", use_container_width=True)
        st.error(
            "I couldn't confidently detect the coloured circle automatically. "
            "Try a brighter, top-down photo with the indicator circle clearly visible against a plain background."
        )
    else:
