import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

st.set_page_config(page_title="Smart Herbal Wound Dressing", page_icon="🩹", layout="centered")

st.title("AI-Assisted Smart Herbal Wound Dressing")
st.write("Capture a clear photo of the indicator patch to measure its visible colour.")
st.warning(
    "This app measures colour only. It does not diagnose a wound, detect infection, "
    "or determine pH unless the indicator has been calibrated experimentally."
)

def detect_indicator(rgb_image):
    """Find a likely pink/purple indicator region near the centre/lower-centre."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Pink, red and purple hues in OpenCV HSV (hue range 0-179).
    mask1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([145, 25, 25]), np.array([179, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([125, 25, 25]), np.array([144, 255, 255]))
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    height, width = mask.shape
    # Focus on the central/lower-central area where the indicator circle is expected.
    roi_y1, roi_y2 = int(height * 0.20), int(height * 0.90)
    roi_x1, roi_x2 = int(width * 0.15), int(width * 0.85)
    roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
    roi = cv2.medianBlur(roi, 5)

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
        # Prefer roughly circular shapes; tolerate imperfect photos.
        score = area * (1.0 - min(abs(1.0 - aspect), 1.0)) * (0.5 + circularity)
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    # Convert ROI coordinates back to full image coordinates.
    x += roi_x1
    y += roi_y1
    return (x, y, w, h)


camera_image = st.camera_input("Open camera and capture the patch")

with st.expander("Or upload a photo instead"):
    uploaded_image = st.file_uploader("Choose a patch photo", type=["png", "jpg", "jpeg"])

image_file = camera_image if camera_image is not None else uploaded_image

if image_file is None:
    st.info("Use the camera above, or open the upload option, to provide a photo.")
else:
    try:
        pil_image = Image.open(image_file).convert("RGB")
        rgb = np.array(pil_image)

        st.subheader("Captured photo")
        st.image(pil_image, use_container_width=True)

        detected_box = detect_indicator(rgb)

        if detected_box is None:
            st.warning(
                "I could not confidently find a pink/purple indicator area. "
                "Try a clear, well-lit photo with the indicator circle fully visible."
            )
        else:
            x, y, w, h = detected_box
            # Draw a visible outline around the detected region.
            outlined = pil_image.copy()
            draw = ImageDraw.Draw(outlined)
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=max(3, rgb.shape[1] // 250))
            st.subheader("Detected indicator area")
            st.image(outlined, caption="Green outline shows the area selected by the app.", use_container_width=True)

            # Sample the central portion of the detected region to reduce edge/background effects.
            pad_x = max(1, int(w * 0.20))
            pad_y = max(1, int(h * 0.20))
            x1, x2 = max(0, x + pad_x), min(rgb.shape[1], x + w - pad_x)
            y1, y2 = max(0, y + pad_y), min(rgb.shape[0], y + h - pad_y)
            crop = rgb[y1:y2, x1:x2]

            if crop.size == 0:
                crop = rgb[y:y + h, x:x + w]

            median_rgb = np.median(crop.reshape(-1, 3), axis=0).astype(int)
            r, g, b = [int(v) for v in median_rgb]
            median_hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
            hh, ss, vv = [int(v) for v in median_hsv]
            hex_colour = f"#{r:02X}{g:02X}{b:02X}"

            st.subheader("Measured colour")
            st.markdown(
                f'<div style="padding:12px;border-radius:8px;background:{hex_colour};'
                f'color:{"white" if (r*0.299+g*0.587+b*0.114)<140 else "black"};'
                f'border:1px solid #888;">Sample colour: {hex_colour}</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RGB", f"{r}, {g}, {b}")
            with col2:
                st.metric("HSV", f"{hh}, {ss}, {vv}")

            st.caption(
                "Colour values depend on lighting, camera settings, and the patch material. "
                "To estimate pH, first build and validate a calibration curve using known pH buffer solutions."
            )
    except Exception as error:
        st.error("The photo could not be processed. Try another clear JPG or PNG photo.")
        st.caption(f"Technical detail: {error}")
