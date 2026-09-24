"""Cassava Leaf Disease Detection Dashboard (PyTorch)."""

import torch
torch.set_num_threads(1)

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from model_utils import (
    load_cassava_model, preprocess_image, predict, generate_gradcam,
)
from disease_info import DISEASE_INFO

MODEL_PATH = "models/cassava_mobilenetv2_final.pth"
METADATA_PATH = "models/cassava_metadata.json"

st.set_page_config(page_title="Cassava Leaf Disease Detector", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 2.4rem; font-weight: 700; color: #1b5e20; text-align: center; }
.subtitle   { text-align: center; color: #555; margin-bottom: 1.5rem; }
.prediction-box { background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 1.2rem; border-radius: 12px; border-left: 6px solid #2e7d32; }
.confidence-box { background: #f1f8e9; padding: 1rem; border-radius: 10px;
    border-left: 6px solid #558b2f; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Cassava Leaf Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a cassava leaf image to detect diseases using AI</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_cassava_model(MODEL_PATH, METADATA_PATH)


try:
    model, class_names, target_layers, img_size = get_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Deep learning classifier with **Grad-CAM** explanations.")
    st.markdown("**Classes:**")
    for c in class_names:
        st.markdown(f"- {c}")

uploaded_file = st.file_uploader("📤 Upload a cassava leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info(" Please upload a cassava leaf image to begin.")
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader(" Uploaded Image")
    st.image(pil_image, use_container_width=True)

with st.spinner("🔬 Analysing..."):
    batch = preprocess_image(pil_image, img_size)
    pred_idx, confidence, all_probs = predict(model, batch)
    predicted_class = class_names[pred_idx]
    try:
        overlay_img = generate_gradcam(model, target_layers, batch, pred_idx, pil_image, img_size)
    except Exception as e:
        overlay_img = None
        st.warning(f"Grad-CAM failed: {e}")

with col_right:
    st.subheader(" Grad-CAM Heatmap")
    if overlay_img is not None:
        st.image(overlay_img, use_container_width=True)
    else:
        st.info("Grad-CAM unavailable.")

st.divider()
st.subheader(" Prediction Result")
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f'<div class="prediction-box"><h3 style="margin:0;color:#1b5e20;">{predicted_class}</h3>'
                '<p style="margin:0.3rem 0 0 0;color:#2e7d32;">Detected disease class</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="confidence-box"><h3 style="margin:0;color:#33691e;">{confidence*100:.1f}%</h3>'
                '<p style="margin:0.3rem 0 0 0;color:#558b2f;">Confidence</p></div>',
                unsafe_allow_html=True)

st.progress(min(confidence, 1.0))

with st.expander("📊 Full class probabilities"):
    fig, ax = plt.subplots(figsize=(8, 3))
    y_pos = np.arange(len(class_names))
    colors = ["#2e7d32" if i == pred_idx else "#a5d6a7" for i in range(len(class_names))]
    ax.barh(y_pos, all_probs, color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    for i, v in enumerate(all_probs):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=8)
    st.pyplot(fig)

st.divider()
st.subheader(" Disease Information & Management")
info = DISEASE_INFO.get(predicted_class)
if info:
    st.markdown(f"**Description:** {info['description']}")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🔍 Symptoms")
        for s in info["symptoms"]: st.markdown(f"- {s}")
    with colB:
        st.markdown("####  Recommended Actions")
        for a in info["advice"]: st.markdown(f"- {a}")

st.caption("⚠️ Decision-support tool only.")
