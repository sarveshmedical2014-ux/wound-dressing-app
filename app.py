import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

st.set_page_config(page_title="Herbal Wound Dressing Colour Analyzer", layout="centered")
st.title("AI-Assisted Smart Herbal Wound Dressing")
st.subheader("Automatic central indicator detection")

st.write(
    "Take a photo of the patch. The app will try to find the pink-purple indicator circle, "
    "outline the selected area, and measure its photographed colour."
)
st.warning(
    "This app measures colour only. It cannot confirm pH or diagnose wound health until "
    "the indicator has been tested against known-pH solutions."
)

st.markdown("### Take a photo")
camera_photo = st.camera_input("Open camera and capture the patch")

st.markdown("Or use an existing photo if needed.")
uploaded_photo = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
photo = camera_photo if camera_photo is not None else uploaded_photo


def detect_indicator(rgb):
    """Find a likely pink/red/purple indicator in the lower-middle area of the photo."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Search in the lower-central portion of the image, matching the example photo framing.
    x_start, x_end = int(width * 0.20), int(width * 0.75)
    y_start, y_end = int(height * 0.40), int(height * 0.95)
    region = hsv[y_start:y_end, x_start:x_end]

    hue = region[:, :, 0]
    saturation = region[:, :, 1]
    value = region[:, :, 2]

    pink_red_purple = (hue <= 12) | ((hue >= 125) & (hue <= 179))
    mask = (pink_red_purple & (saturation >= 40) & (value >= 20)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_area = region.shape[0] * region.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(60, region_area * 0.0008) or area > region_area * 0.30:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 6:
            continue

        target_x = region.shape[1] * 0.55
        target_y = region.shape[0] * 0.70
        distance = np.sqrt(
            ((cx - target_x) / max(1, region.shape[1])) ** 2
            + ((cy - target_y) / max(1, region.shape[0])) ** 2
        )
        score = area * (0.35 + min(circularity, 1.0)) / (1 + 5 * distance)
        candidates.append((score, contour, circularity))

    if not candidates:
        return None

    _, contour, circularity = max(candidates, key=lambda item: item[0])
    x, y, box_width, box_height = cv2.boundingRect(contour)
    margin = max(3, int(max(box_width, box_height) * 0.08))

    x1 = max(0, x_start + x - margin)
    y1 = max(0, y_start + y - margin)
    x2 = min(width, x_start + x + box_width + margin)
    y2 = min(height, y_start + y + box_height + margin)
    return x1, y1, x2, y2, circularity


if photo is None:
    st.info("Use the camera above to take a photo. The result will appear below the captured image.")
else:
    try:
        image = Image.open(photo).convert("RGB")
        rgb = np.array(image)
        result = detect_indicator(rgb)
