import streamlit as st
import cv2
import numpy as np
from PIL import Image

# =======================================================
# 1. YOUR LAB CALIBRATION DATA
# =======================================================
# Enter your exact RGB values next to each specific pH number.
# Keep them sorted from lowest pH to highest pH!
CALIBRATION_POINTS = [
    {"ph": 5.5, "rgb": [120, 180, 95]},  # Replace with actual RGB for pH 5.5
    {"ph": 6.5, "rgb": [170, 190, 80]},  # Replace with actual RGB for pH 6.5
    {"ph": 7.5, "rgb": [210, 150, 70]},  # Replace with actual RGB for pH 7.5
    {"ph": 8.5, "rgb":}    # Replace with actual RGB for pH 8.5
]

def predict_precise_ph(target_rgb):
    """Calculates the exact decimal pH using color distance interpolation."""
    target = np.array(target_rgb)
    distances = []
    
    for pt in CALIBRATION_POINTS:
        cal_rgb = np.array(pt["rgb"])
        dist = np.linalg.norm(target - cal_rgb)
        distances.append((dist, pt["ph"]))
    
    distances.sort(key=lambda x: x[0])
    closest = distances[0]
    second_closest = distances[1]
    
    d1, ph1 = closest
    d2, ph2 = second_closest
    
    if d1 + d2 == 0:
        return ph1
        
    weight1 = d2 / (d1 + d2)
    weight2 = d1 / (d1 + d2)
    precise_ph = (ph1 * weight1) + (ph2 * weight2)
    return round(precise_ph, 2)

def get_clinical_prediction(ph):
    """Returns the medical prediction based on the calculated pH scale."""
    if ph <= 6.0:
        return "💚 HEALTHY: Normal acidic skin environment. The herbal dressing is maintaining a strong healing barrier.", "success"
    elif 6.0 < ph <= 7.0:
        return "💛 HEALING PROGRESS: Wound tissue is actively regenerating. Continue monitoring.", "info"
    elif 7.0 < ph <= 8.0:
        return "⚠️ WARNING (HIGH RISK): Early chemical shift detected. High risk of bacterial growth or inflammation.", "warning"
    else:
        return "🚨 CRITICAL ALARM: Active infection detected! Alkaline spike indicates high bacterial colonization. Change dressing immediately.", "error"

# =======================================================
# 2. SMARTPHONE INTERFACE DESIGN
# =======================================================
st.set_page_config(page_title="Smart Wound Monitor", page_icon="🩹", layout="centered")

st.title("🩹 AI-Assisted Smart Wound Monitor")
st.write("Snap a photo of the patch. The AI will calculate the exact pH and predict wound status.")

uploaded_file = st.camera_input("Capture Dressing Patch")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    avg_color = cv2.mean(img_np)[:3]
    avg_rgb = [int(x) for x in avg_color]
    
    calculated_ph = predict_precise_ph(avg_rgb)
    message, status_type = get_clinical_prediction(calculated_ph)
    
    st.divider()
    st.subheader("AI Analysis Results")
    st.metric(label="Calculated Wound pH Level", value=f"pH {calculated_ph}")
    
    color_hex = '#{:02x}{:02x}{:02x}'.format(*avg_rgb)
    st.write("**Detected Patch Color:**")
    st.markdown(f'<div style="background-color:{color_hex}; width:100%; height:40px; border-radius:8px; border:1px solid #ddd; margin-bottom:15px;"></div>', unsafe_allow_html=True)
    
    if status_type == "success":
        st.success(message)
    elif status_type == "info":
        st.info(message)
    elif status_type == "warning":
        st.warning(message)
    else:
        st.error(message)
