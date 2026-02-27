"""
app.py — Streamlit web application for Tamil Handwritten Text Extraction.

Supports:
  - EasyOCR (primary, works immediately)
  - Custom CNN (optional, needs training)
  - MobileNetV2 (optional, needs training)

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import io

BACKEND_URL = "http://localhost:8000"


# Page configuration
st.set_page_config(
    page_title="Tamil Handwritten Text Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .result-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }

    .result-text {
        font-size: 1.8rem;
        color: #e2e8f0;
        line-height: 2;
    }

    .stat-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .model-badge-easyocr {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        color: #1a1a2e;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .model-badge-cnn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ────────────────────────────────────────────────────────
    st.markdown('<h1 class="hero-title">✍️ Tamil Handwritten Text Extractor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered recognition of handwritten Tamil text from images</p>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="Minimum confidence for text prediction"
        )
        show_segmentation = st.checkbox("Show Detection Boxes", value=True,
                                         help="Display bounding boxes on detected text")
        show_details = st.checkbox("Show Prediction Details", value=False,
                                    help="Show confidence for each detected region")

        st.markdown("---")
        st.markdown("### 🧠 Model Selection")

        from src.extractor import get_available_models
        available_models = get_available_models()
        model_names = list(available_models.keys())

        selected_model_name = st.selectbox(
            "Choose Model",
            options=model_names,
            help="EasyOCR works immediately. CNN/MobileNet need training first."
        )
        selected_model_info = available_models[selected_model_name]

        if selected_model_info['type'] == 'easyocr':
            st.markdown('<span class="model-badge-easyocr">✅ READY — No training needed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="model-badge-cnn">🧠 Trained Model</span>', unsafe_allow_html=True)

        # Show comparison results if available
        comparison_path = 'models/comparison_results.json'
        if os.path.exists(comparison_path):
            st.markdown("---")
            st.markdown("### 📊 Model Comparison")
            import json
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
            for name, data in comparison.items():
                best_acc = max(d['test_accuracy'] for d in comparison.values())
                emoji = "🏆" if data['test_accuracy'] == best_acc else "📦"
                st.markdown(f"{emoji} **{name}**: {data['test_accuracy']}%")

        st.markdown("---")
        st.markdown("### 📝 About")
        st.markdown("""
        **Models available:**
        - 🟢 **EasyOCR** — Pre-trained, instant
        - 🔵 **Custom CNN** — Train with dataset
        - 🟣 **MobileNetV2** — Transfer learning
        """)

        st.markdown("---")
        st.markdown("### 🔌 Backend Mode")
        use_api = st.checkbox("Use FastAPI Backend", value=False,
                              help="Call FastAPI backend at localhost:8000 instead of direct extraction")
        if use_api:
            try:
                r = requests.get(f"{BACKEND_URL}/api/health", timeout=2)
                if r.status_code == 200:
                    st.markdown('<span class="model-badge-easyocr">✅ Backend Connected</span>', unsafe_allow_html=True)
                else:
                    st.warning("Backend returned error")
                    use_api = False
            except Exception:
                st.warning("Backend not running. Start it with:\n`python backend.py`")
                use_api = False

    # ── Main Content ──────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a handwritten Tamil text image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image containing handwritten Tamil text"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-label">
                    📐 {image.size[0]} × {image.size[1]} px &nbsp;|&nbsp;
                    📁 {uploaded_file.size / 1024:.1f} KB &nbsp;|&nbsp;
                    🎨 {image.mode}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📝 Extracted Text")

        if uploaded_file is not None:
            extract_btn = st.button("🚀 Extract Text", use_container_width=True)

            if extract_btn:
                with st.spinner(f"🔄 Extracting with {selected_model_name}..."):
                    try:
                        text = ""
                        details = []

                        if use_api:
                            # ── API Mode: call FastAPI backend ──
                            uploaded_file.seek(0)
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            params = {
                                "model": selected_model_info['type'],
                                "confidence_threshold": confidence_threshold
                            }
                            resp = requests.post(
                                f"{BACKEND_URL}/api/extract",
                                files=files,
                                params=params,
                                timeout=60
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                text = data['text']
                                details = [
                                    {'label': r['label'], 'confidence': r['confidence'], 'bbox': tuple(r['bbox'])}
                                    for r in data['regions']
                                ]
                            else:
                                st.error(f"API Error: {resp.json().get('error', resp.text)}")

                        else:
                            # ── Direct Mode: load model locally ──
                            from src.extractor import load_extractor
                            extractor = load_extractor(selected_model_info)

                            img_array = np.array(image)
                            if len(img_array.shape) == 2:
                                img_cv = img_array
                            elif img_array.shape[2] == 4:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                            else:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                            temp_path = 'temp_upload.png'
                            cv2.imwrite(temp_path, img_cv)
                            text, details = extractor.predict_text(
                                temp_path, confidence_threshold=confidence_threshold
                            )
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                        # Display results
                        if text:
                            st.markdown(f"""
                            <div class="result-box">
                                <div class="result-text">{text}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Stats
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-number">{len(details)}</div>
                                    <div class="stat-label">Regions Detected</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with c2:
                                avg_conf = np.mean([d['confidence'] for d in details]) if details else 0
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-number">{avg_conf:.1%}</div>
                                    <div class="stat-label">Avg Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with c3:
                                high_conf = sum(1 for d in details if d['confidence'] > 0.8)
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-number">{high_conf}</div>
                                    <div class="stat-label">High Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)

                            # Download
                            st.download_button(
                                label="📥 Download Extracted Text",
                                data=text,
                                file_name="extracted_tamil_text.txt",
                                mime="text/plain"
                            )

                            # Details
                            if show_details and details:
                                st.markdown("### 🔍 Detection Details")
                                for i, d in enumerate(details):
                                    conf_color = "🟢" if d['confidence'] > 0.8 else "🟡" if d['confidence'] > 0.5 else "🔴"
                                    st.markdown(
                                        f"{conf_color} **Region {i+1}**: `{d['label']}` "
                                        f"— Confidence: {d['confidence']:.2%}"
                                    )

                        else:
                            st.warning("No text detected. Try a clearer image or lower the confidence threshold.")

                        # Segmentation visualization
                        if show_segmentation and details:
                            st.markdown("### ✂️ Text Detection")
                            img_draw = np.array(image.copy())
                            for d in details:
                                x, y, w, h = d['bbox']
                                color = (0, 255, 0) if d['confidence'] > 0.5 else (255, 165, 0)
                                cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, 2)
                            st.image(img_draw, caption="Detected Regions",
                                    use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
        else:
            st.markdown("""
            <div class="result-box" style="text-align: center; padding: 60px 20px;">
                <p style="font-size: 3rem; margin: 0;">📄</p>
                <p style="color: #a0aec0; font-size: 1.1rem;">
                    Upload an image to get started
                </p>
                <p style="color: #718096; font-size: 0.9rem;">
                    Supported: JPG, PNG, BMP, TIFF
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.85rem;">
        <p>✍️ AI-Powered Tamil Handwritten Text Extractor | EasyOCR + CNN + MobileNetV2 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
