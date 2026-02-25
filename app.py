"""
app.py — Streamlit web application for Tamil Handwritten Text Extraction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

# Page configuration — must be first Streamlit command
st.set_page_config(
    page_title="Tamil Handwritten Text Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for premium dark theme ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }

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
        direction: ltr;
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

    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(102, 126, 234, 0.05);
    }

    .char-prediction {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        text-align: center;
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

    .sidebar .sidebar-content {
        background: rgba(15, 12, 41, 0.95);
    }
</style>
""", unsafe_allow_html=True)


def load_extractor(model_path):
    """Load the Tamil text extractor model."""
    from src.extractor import TamilTextExtractor

    mapping_path = 'models/label_mapping.json'

    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        return None

    return TamilTextExtractor(model_path, mapping_path)


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
            help="Minimum confidence for character prediction"
        )
        show_segmentation = st.checkbox("Show Segmentation", value=True,
                                         help="Display character bounding boxes")
        show_details = st.checkbox("Show Prediction Details", value=False,
                                    help="Show confidence for each character")

        st.markdown("---")
        st.markdown("### 🧠 Model Selection")

        from src.extractor import get_available_models
        available_models = get_available_models()

        if available_models:
            selected_model_name = st.selectbox(
                "Choose Model",
                options=list(available_models.keys()),
                help="Select which trained model to use for extraction"
            )
            selected_model_path = available_models[selected_model_name]
            model_exists = True

            st.success(f"✅ {selected_model_name}")
            mapping_path = 'models/label_mapping.json'
            if os.path.exists(mapping_path):
                import json
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                st.info(f"📝 {len(mapping)} character classes")

            # Show comparison results if available
            comparison_path = 'models/comparison_results.json'
            if os.path.exists(comparison_path):
                st.markdown("---")
                st.markdown("### 📊 Model Comparison")
                import json
                with open(comparison_path, 'r') as f:
                    comparison = json.load(f)
                for name, data in comparison.items():
                    emoji = "🏆" if data['test_accuracy'] == max(d['test_accuracy'] for d in comparison.values()) else "📦"
                    st.markdown(f"{emoji} **{name}**: {data['test_accuracy']}%")

            # Show comparison chart if available
            comparison_img = 'models/model_comparison.png'
            if os.path.exists(comparison_img):
                st.image(comparison_img, caption="Accuracy Comparison", use_container_width=True)
        else:
            selected_model_path = None
            model_exists = False
            st.error("❌ No trained models found")
            st.markdown("""
            Train both models:
            ```
            python -m src.train --model_type both
            ```
            """)

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [Project README](./README.md)")
        st.markdown("- [Dataset Info](./data/README.md)")

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
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Image info
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
                if not model_exists or not selected_model_path:
                    st.error("⚠️ Please train the model first before extracting text.")
                    st.code("python -m src.train --model_type both", language="bash")
                else:
                    with st.spinner(f"🔄 Extracting with {selected_model_name}..."):
                        try:
                            extractor = load_extractor(selected_model_path)
                            if extractor is None:
                                st.error("Failed to load model.")
                                return

                            # Convert PIL to OpenCV format
                            img_array = np.array(image)
                            if len(img_array.shape) == 2:
                                img_cv = img_array
                            else:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                            # Save temp image for extraction
                            temp_path = 'temp_upload.png'
                            cv2.imwrite(temp_path, img_cv)

                            # Extract text
                            text, details = extractor.predict_text(
                                temp_path,
                                confidence_threshold=confidence_threshold
                            )

                            # Clean up
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                            # Display results
                            if text:
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-text">{text}</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Stats row
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="stat-number">{len(details)}</div>
                                        <div class="stat-label">Characters Found</div>
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

                                # Download button
                                st.download_button(
                                    label="📥 Download Extracted Text",
                                    data=text,
                                    file_name="extracted_tamil_text.txt",
                                    mime="text/plain"
                                )

                                # Show detailed predictions
                                if show_details and details:
                                    st.markdown("### 🔍 Character Details")
                                    for i, d in enumerate(details):
                                        conf_color = "🟢" if d['confidence'] > 0.8 else "🟡" if d['confidence'] > 0.5 else "🔴"
                                        st.markdown(
                                            f"{conf_color} **Char {i+1}**: `{d['label']}` "
                                            f"— Confidence: {d['confidence']:.2%}"
                                        )

                            else:
                                st.warning("No characters detected in the image. "
                                          "Try adjusting the confidence threshold or "
                                          "upload a clearer image.")

                            # Show segmentation visualization
                            if show_segmentation and details:
                                st.markdown("### ✂️ Character Segmentation")
                                annotated = image.copy()
                                img_draw = np.array(annotated)
                                for i, d in enumerate(details):
                                    x, y, w, h = d['bbox']
                                    color = (0, 255, 0) if d['confidence'] > 0.5 else (255, 165, 0)
                                    cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, 2)
                                st.image(img_draw, caption="Segmented Characters",
                                        use_container_width=True)

                        except Exception as e:
                            st.error(f"Error during extraction: {str(e)}")
                            st.exception(e)
        else:
            st.markdown("""
            <div class="result-box" style="text-align: center; padding: 60px 20px;">
                <p style="font-size: 3rem; margin: 0;">📄</p>
                <p style="color: #a0aec0; font-size: 1.1rem;">
                    Upload an image to get started
                </p>
                <p style="color: #718096; font-size: 0.9rem;">
                    Supported formats: JPG, PNG, BMP, TIFF
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.85rem;">
        <p>✍️ AI-Powered Tamil Handwritten Text Extractor | Built with TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
