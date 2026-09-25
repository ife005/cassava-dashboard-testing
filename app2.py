"""Cassava Leaf Disease Detection Dashboard (PyTorch) — Full Version."""

import os
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

# ============================================================
#  CONFIG
# ============================================================
MODEL_PATH = "models/cassava_mobilenetv2_final.pth"
METADATA_PATH = "models/cassava_metadata.json"

st.set_page_config(
    page_title="Cassava Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-title { font-size: 2.4rem; font-weight: 700; color: #1b5e20; text-align: center; }
.subtitle   { text-align: center; color: #555; margin-bottom: 1.5rem; }
.prediction-box { background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 1.2rem; border-radius: 12px; border-left: 6px solid #2e7d32; }
.confidence-box { background: #f1f8e9; padding: 1rem; border-radius: 10px;
    border-left: 6px solid #558b2f; }
.info-card { background: #f8f9fa; padding: 1rem; border-radius: 10px;
    border: 1px solid #dee2e6; }
.warn-card { background: #fff3cd; padding: 1rem; border-radius: 10px;
    border-left: 6px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Cassava Leaf Disease Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload, paste (Ctrl+V), or capture a cassava leaf image to detect diseases with AI</div>',
    unsafe_allow_html=True,
)

# ============================================================
#  MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_cassava_model(MODEL_PATH, METADATA_PATH)

try:
    model, class_names, target_layers, img_size = get_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ============================================================
#  SESSION STATE
# ============================================================
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Deep learning classifier trained on cassava leaf images. "
        "Uses **MobileNetV2** + **Grad-CAM** for visual explanations."
    )

    st.divider()
    st.markdown("### 🌱 Diseases Covered")
    for c in class_names:
        st.markdown(f"- {c}")

    st.divider()
    st.markdown("### 📊 Model Info")
    st.markdown(
        """
        - **Architecture**: MobileNetV2
        - **Input size**: 224×224
        - **Classes**: 5
        - **Framework**: PyTorch
        """
    )

    st.divider()
    if st.button("🔄 Clear current image", use_container_width=True):
        st.session_state.current_image = None
        st.rerun()

    st.divider()
    st.caption("⚠️ Decision-support tool only. Always confirm with a local agricultural extension officer.")

# ============================================================
#  HOW IT WORKS
# ============================================================
with st.expander("📚 How does this work?"):
    st.markdown(
        """
        **1. Image preprocessing**
        Your image is resized to 224×224 pixels and normalized before being fed to the model.

        **2. MobileNetV2 classifier**
        A lightweight CNN pretrained on ImageNet, fine-tuned to recognize 5 cassava leaf classes.
        It runs efficiently on CPU.

        **3. Softmax output**
        The model outputs a probability for each class. The highest becomes the prediction.

        **4. Grad-CAM heatmap**
        We visualize **which regions** of the image most influenced the prediction.
        Warm colours (red) = high attention. Cool colours (blue) = low attention.
        This helps you verify the model is looking at actual leaf symptoms, not background noise.
        """
    )

# ============================================================
#  INPUT TABS
# ============================================================
tab_upload, tab_samples, tab_paste, tab_camera = st.tabs(
    ["📤 Upload", "🖼️ Sample Images", "📋 Paste (Ctrl+V)", "📷 Camera"]
)

# ---------- Tab 1: Upload ----------
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a cassava leaf image",
        type=["jpg", "jpeg", "png"],
        key="upload_tab",
    )
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file).convert("RGB")
        st.success("✅ Image uploaded!")

# ---------- Tab 2: Sample images ----------
with tab_samples:
    st.caption("Don't have an image? Try one of these examples.")

    SAMPLE_DIR = "samples"
    SAMPLE_FILES = {
        "CBB": f"{SAMPLE_DIR}/cbb.jpg",
        "CBSD": f"{SAMPLE_DIR}/cbsd.jpg",
        "CGMD": f"{SAMPLE_DIR}/cgmd.jpg",
        "CMD": f"{SAMPLE_DIR}/cmd.jpg",
        "Healthy": f"{SAMPLE_DIR}/healthy.jpg",
    }

    sample_cols = st.columns(5)
    for col, (name, path) in zip(sample_cols, SAMPLE_FILES.items()):
        with col:
            if os.path.exists(path):
                st.image(path, caption=name, use_container_width=True)
                if st.button(f"Use {name}", key=f"use_{name}", use_container_width=True):
                    st.session_state.current_image = Image.open(path).convert("RGB")
                    st.rerun()
            else:
                st.info(f"Missing sample: {name}")

    st.info(
        "📌 To enable samples, add JPG files to a `samples/` folder in your repo: "
        "`cbb.jpg`, `cbsd.jpg`, `cgmd.jpg`, `cmd.jpg`, `healthy.jpg`."
    )

# ---------- Tab 3: Clipboard paste ----------
with tab_paste:
    st.caption(
        "**Paste an image directly from your clipboard.** "
        "Take a screenshot or copy an image from anywhere, then press **Ctrl+V** (Cmd+V on Mac) below."
    )
    pasted = st.file_uploader(
        "Paste here (Ctrl+V)",
        type=["png", "jpg", "jpeg"],
        key="paste_tab",
        label_visibility="collapsed",
    )
    if pasted is not None:
        st.session_state.current_image = Image.open(pasted).convert("RGB")
        st.success("✅ Image pasted from clipboard!")

# ---------- Tab 4: Live camera capture ----------
with tab_camera:
    st.caption(
        "**Capture a cassava leaf with your device's camera.** "
        "Works on phones, tablets, and laptops with a webcam."
    )

    cam_file = st.camera_input(
        "Point your camera at a cassava leaf",
        key="camera_tab_input",
    )

    if cam_file is not None:
        st.session_state.current_image = Image.open(cam_file).convert("RGB")
        st.success("✅ Photo captured! Scroll down for the analysis.")

    st.info(
        "📱 **Tip for farmers**: On your phone, hold the leaf steady in good lighting. "
        "Make sure the affected area fills the frame."
    )

# ============================================================
#  MAIN ANALYSIS
# ============================================================
pil_image = st.session_state.current_image

if pil_image is None:
    st.info("👆 Upload, drag, paste (Ctrl+V), or capture a photo to begin.")
    st.stop()

st.divider()
st.subheader("🔬 Analysis")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🖼️ Your Image")
    st.image(pil_image, use_container_width=True)

with st.spinner("🔬 Analysing..."):
    batch = preprocess_image(pil_image, img_size)
    pred_idx, confidence, all_probs = predict(model, batch)
    predicted_class = class_names[pred_idx]
    try:
        overlay_img = generate_gradcam(
            model, target_layers, batch, pred_idx, pil_image, img_size
        )
    except Exception as e:
        overlay_img = None
        st.warning(f"Grad-CAM failed: {e}")

with col_right:
    st.markdown("#### 🔥 Grad-CAM Heatmap")
    if overlay_img is not None:
        st.image(overlay_img, use_container_width=True)
        st.caption("Warm regions = areas the model focused on when making this prediction.")
    else:
        st.info("Grad-CAM unavailable.")

# ============================================================
#  PREDICTION RESULT
# ============================================================
st.divider()
st.subheader("🧪 Prediction Result")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        f'<div class="prediction-box"><h3 style="margin:0;color:#1b5e20;">{predicted_class}</h3>'
        '<p style="margin:0.3rem 0 0 0;color:#2e7d32;">Detected disease class</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="confidence-box"><h3 style="margin:0;color:#33691e;">{confidence*100:.1f}%</h3>'
        '<p style="margin:0.3rem 0 0 0;color:#558b2f;">Confidence</p></div>',
        unsafe_allow_html=True,
    )

st.progress(min(confidence, 1.0))

# ---------- Confidence interpretation ----------
if confidence >= 0.85:
    st.success(
        f"**High confidence ({confidence*100:.1f}%)** — The model is very sure. "
        "You can rely on this result for decision-making, but still verify visually."
    )
elif confidence >= 0.60:
    st.warning(
        f"**Medium confidence ({confidence*100:.1f}%)** — The model is fairly sure but not certain. "
        "Cross-check with the Grad-CAM heatmap and consult an extension officer."
    )
else:
    st.error(
        f"**Low confidence ({confidence*100:.1f}%)** — The model is unsure. "
        "The image may be unclear, contain multiple leaves, or show early-stage symptoms. "
        "Please try a clearer photo."
    )

# ---------- Probability chart ----------
with st.expander("📊 Full class probabilities"):
    fig, ax = plt.subplots(figsize=(8, 3))
    y_pos = np.arange(len(class_names))
    colors = ["#2e7d32" if i == pred_idx else "#a5d6a7" for i in range(len(class_names))]
    ax.barh(y_pos, all_probs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for i, v in enumerate(all_probs):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=8)
    st.pyplot(fig)

# ============================================================
#  DISEASE INFO & MANAGEMENT
# ============================================================
st.divider()
st.subheader("📖 Disease Information & Management")

info = DISEASE_INFO.get(predicted_class)
if info:
    st.markdown(f"**Description:** {info['description']}")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🔍 Symptoms")
        for s in info["symptoms"]:
            st.markdown(f"- {s}")
    with colB:
        st.markdown("#### ✅ Recommended Actions")
        for a in info["advice"]:
            st.markdown(f"- {a}")
else:
    st.info("No additional information available for this class.")

# ============================================================
#  WHAT TO DO NEXT
# ============================================================
st.divider()
st.markdown("### 🚜 What to do next")
st.markdown(
    """
    - **Isolate** the affected plants if possible to prevent spread.
    - **Photograph** the leaf and plant from multiple angles for records.
    - **Contact** your local agricultural extension officer for confirmation.
    - **Share** this result with neighbouring farmers so they can scout their fields.
    """
)

st.caption("⚠️ Decision-support tool only. Not a substitute for professional diagnosis.")
