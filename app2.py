"""Cassava Leaf Disease Detection Dashboard (PyTorch) — Full Version with PDF + Batch."""

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
from report_generator import build_pdf_report

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
    '<div class="subtitle">Upload, paste, capture, or batch-process cassava leaves to detect diseases with AI</div>',
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
    st.caption("⚠️ Decision-support tool only.")

# ============================================================
#  HOW IT WORKS
# ============================================================
with st.expander("📚 How does this work?"):
    st.markdown(
        """
        **1. Image preprocessing** — Your image is resized to 224×224 and normalized.

        **2. MobileNetV2 classifier** — A lightweight CNN pretrained on ImageNet, fine-tuned for 5 cassava classes.

        **3. Softmax output** — Highest probability becomes the prediction.

        **4. Grad-CAM heatmap** — Shows which regions influenced the prediction (red = high attention).
        """
    )

# ============================================================
#  INPUT TABS
# ============================================================
tab_upload, tab_samples, tab_paste, tab_camera, tab_batch = st.tabs(
    ["📤 Upload", "🖼️ Sample Images", "📋 Paste (Ctrl+V)", "📷 Camera", "📦 Batch Mode"]
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

# ---------- Tab 3: Paste ----------
with tab_paste:
    st.caption(
        "**Paste an image directly from your clipboard.** "
        "Take a screenshot or copy an image, then press **Ctrl+V** (Cmd+V on Mac)."
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

# ---------- Tab 4: Camera ----------
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
        st.success("✅ Photo captured!")
    st.info("📱 **Tip**: Hold the leaf steady in good lighting. Fill the frame with the affected area.")

# ---------- Tab 5: Batch Mode ----------
with tab_batch:
    st.caption(
        "**Upload multiple leaf images at once.** "
        "Great for surveying a whole field — get a summary table of all predictions."
    )
    batch_files = st.file_uploader(
        "Upload up to 20 images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_tab",
    )

    if batch_files:
        if len(batch_files) > 20:
            st.warning("⚠️ Limited to first 20 images.")
            batch_files = batch_files[:20]

        st.success(f"✅ {len(batch_files)} image(s) loaded. Running batch analysis...")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(batch_files):
            status_text.text(f"Analysing image {i+1} of {len(batch_files)}: {file.name}")
            try:
                img = Image.open(file).convert("RGB")
                b = preprocess_image(img, img_size)
                idx, conf, _ = predict(model, b)
                results.append({
                    "File": file.name,
                    "Prediction": class_names[idx],
                    "Confidence": f"{conf*100:.1f}%",
                    "_confidence_float": conf,
                    "_pil": img,
                })
            except Exception as e:
                results.append({
                    "File": file.name,
                    "Prediction": f"❌ Error",
                    "Confidence": "-",
                    "_confidence_float": 0.0,
                    "_pil": None,
                })
            progress_bar.progress((i + 1) / len(batch_files))

        status_text.text("✅ Batch analysis complete!")
        progress_bar.empty()

        # ---- Summary table ----
        st.subheader("📊 Batch Summary")

        # Class distribution
        from collections import Counter
        counts = Counter(r["Prediction"] for r in results if r["_pil"] is not None)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total images", len(results))
        with col_b:
            most_common = counts.most_common(1)
            if most_common:
                st.metric("Most common", most_common[0][0].split("(")[0].strip())
        with col_c:
            avg_conf = np.mean([r["_confidence_float"] for r in results if r["_pil"]])
            st.metric("Avg confidence", f"{avg_conf*100:.1f}%")

        # Table of results
        import pandas as pd
        df = pd.DataFrame([
            {"File": r["File"], "Prediction": r["Prediction"], "Confidence": r["Confidence"]}
            for r in results
        ])
        st.dataframe(df, use_container_width=True)

        # ---- CSV download ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download results as CSV",
            data=csv,
            file_name="cassava_batch_results.csv",
            mime="text/csv",
        )

        # ---- Image grid ----
        st.subheader("🖼️ Image Gallery")
        st.caption("Click any image to zoom.")
        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            row = results[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, row):
                with col:
                    if r["_pil"] is not None:
                        st.image(r["_pil"], caption=f"{r['File']}\n{r['Prediction']} ({r['Confidence']})", use_container_width=True)

        # ---- Detailed single analysis ----
        st.divider()
        st.subheader("🔍 View detailed analysis for one image")
        selected = st.selectbox(
            "Choose an image:",
            options=[r["File"] for r in results if r["_pil"] is not None],
        )
        if selected:
            st.session_state.current_image = next(
                r["_pil"] for r in results if r["File"] == selected
            )
            st.info("⬇️ Scroll down to see the full analysis of this image.")

# ============================================================
#  SINGLE-IMAGE ANALYSIS
# ============================================================
pil_image = st.session_state.current_image

if pil_image is None:
    st.info("👆 Upload, drag, paste, capture, or batch-process images to begin.")
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
        st.caption("Warm regions = areas the model focused on.")
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

# Confidence interpretation
if confidence >= 0.85:
    st.success(f"**High confidence ({confidence*100:.1f}%)** — Reliable for decision-making.")
elif confidence >= 0.60:
    st.warning(f"**Medium confidence ({confidence*100:.1f}%)** — Cross-check with the heatmap.")
else:
    st.error(f"**Low confidence ({confidence*100:.1f}%)** — Try a clearer photo.")

# Probability chart
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
#  PDF REPORT DOWNLOAD
# ============================================================
st.divider()
st.subheader("📄 Download Report")

st.caption("Generate a PDF report with the prediction, heatmap, and management advice.")

if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
    with st.spinner("Generating PDF..."):
        try:
            pdf_buffer = build_pdf_report(
                original_image=pil_image,
                gradcam_image=overlay_img,
                predicted_class=predicted_class,
                confidence=confidence,
                all_probs=all_probs,
                class_names=class_names,
                disease_info=DISEASE_INFO.get(predicted_class),
            )
            st.download_button(
                label="⬇️ Click here to download the PDF",
                data=pdf_buffer,
                file_name=f"cassava_report_{predicted_class.split('(')[0].strip().replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("✅ PDF ready! Click the button above to download.")
        except Exception as e:
            st.error(f"❌ PDF generation failed: {e}")

# ============================================================
#  DISEASE INFO
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
    st.info("No additional information available.")

st.divider()
st.markdown("### 🚜 What to do next")
st.markdown(
    """
    - **Isolate** affected plants to prevent spread.
    - **Photograph** from multiple angles for records.
    - **Contact** your local agricultural extension officer.
    - **Share** this result with neighbouring farmers.
    """
)

st.caption("⚠️ Decision-support tool only.")
