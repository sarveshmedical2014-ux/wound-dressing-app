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

uploaded_file = st.file_uploader("Upload patch photograph", type=["jpg", "jpeg", "png"])

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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    result = detect_indicator(rgb)

    if result is None:
        st.image(image, caption="Uploaded photograph", use_container_width=True)
        st.error(
            "I couldn't confidently detect the coloured circle automatically. "
            "Try a brighter, top-down photo with the indicator circle clearly visible against a plain background."
        )
    else:
        x1, y1, x2, y2, circularity = result
        roi = rgb[y1:y2, x1:x2]

        # Show the selected region outlined on the full photo.
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=max(3, rgb.shape[1] // 300))
        st.image(preview, caption="Automatically selected indicator region (green outline)", use_container_width=True)
        st.image(roi, caption="Selected central colour region", use_container_width=False)

        # Use pixels that are actually pink/red/purple inside the selected region,
        # avoiding transparent/background pixels included by the rectangular crop.
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hh, ss, vv = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]
        colour_pixels = (((hh <= 12) | ((hh >= 135) & (hh <= 179))) &
                         (ss >= 45) & (vv >= 25) & (vv <= 235))
        pixels = roi[colour_pixels]

        if len(pixels) < 20:
            pixels = roi.reshape(-1, 3)

        median_rgb = tuple(int(v) for v in np.median(pixels, axis=0))
        median_hsv = tuple(int(v) for v in np.median(
            cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3),
            axis=0
        ))
        hex_colour = "#{:02X}{:02X}{:02X}".format(*median_rgb)

        st.markdown("### Measured colour")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:14px;">
              <div style="width:72px;height:72px;border-radius:12px;border:1px solid #777;background:{hex_colour};"></div>
              <div><b>{hex_colour}</b><br>Median RGB: {median_rgb}<br>Median HSV: {median_hsv}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Detection shape score (circularity): {circularity:.2f}")
        st.info(
            "Detection is automatic but not guaranteed. Check the green outline before using the colour values. "
            "These values describe the photo under its current lighting; they do not establish pH."
        )
else:
    st.caption("Upload your photograph to detect and measure the central coloured circle.")
