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

BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Tamil Text Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap');

    * { font-family: 'Outfit', sans-serif; }

    /* ─── Dark Theme ─── */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #111128 40%, #0d1b2a 70%, #0a0a1a 100%);
    }

    /* ─── Hero Header ─── */
    .hero-container {
        text-align: center;
        padding: 20px 0 10px;
    }
    .hero-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 8px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
        font-weight: 400;
    }

    /* ─── Glass Cards ─── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08);
    }
    .glass-card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ─── Result Display ─── */
    .result-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(168, 85, 247, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 28px;
        margin: 8px 0;
    }
    .result-text {
        font-family: 'Noto Sans Tamil', 'Outfit', sans-serif;
        font-size: 1.5rem;
        color: #f1f5f9;
        line-height: 2.2;
        letter-spacing: 0.5px;
    }

    /* ─── Stats Pills ─── */
    .stats-row {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .stat-pill {
        flex: 1;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 12px;
        text-align: center;
        transition: all 0.3s;
    }
    .stat-pill:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #818cf8;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ─── Upload Area ─── */
    .upload-zone {
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        transition: all 0.3s;
        background: rgba(99, 102, 241, 0.02);
    }
    .upload-zone:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.05);
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 8px; }
    .upload-text { color: #94a3b8; font-size: 0.95rem; }
    .upload-hint { color: #475569; font-size: 0.8rem; margin-top: 4px; }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 14px 32px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.35);
    }

    .stDownloadButton > button {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34d399 !important;
        border-radius: 12px;
    }
    .stDownloadButton > button:hover {
        background: rgba(16, 185, 129, 0.25) !important;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #111128 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-top: 20px;
    }

    /* ─── Badges ─── */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-train {
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .badge-api {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
    }

    /* ─── Region Tags ─── */
    .region-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 14px;
        margin: 4px 2px;
        font-size: 0.88rem;
        transition: all 0.2s;
    }
    .region-tag:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .region-word {
        font-family: 'Noto Sans Tamil', sans-serif;
        color: #e2e8f0;
        font-weight: 500;
    }
    .region-conf {
        color: #64748b;
        font-size: 0.75rem;
    }
    .conf-high { color: #34d399; }
    .conf-mid { color: #fbbf24; }
    .conf-low { color: #f87171; }

    /* ─── Image Info Bar ─── */
    .info-bar {
        display: flex;
        justify-content: center;
        gap: 20px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 10px 16px;
        margin-top: 8px;
    }
    .info-item {
        color: #64748b;
        font-size: 0.8rem;
    }
    .info-item span { color: #94a3b8; font-weight: 500; }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #334155;
        font-size: 0.78rem;
        padding: 20px;
        margin-top: 40px;
    }
    .footer a { color: #6366f1; text-decoration: none; }

    /* ─── Hide Streamlit defaults ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Hero Header ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <span class="hero-icon">✍️</span>
        <h1 class="hero-title">Tamil Text Extractor</h1>
        <p class="hero-subtitle">Upload a handwritten Tamil image → Get extracted text instantly</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Controls")

        confidence_threshold = st.slider(
            "Confidence", 0.0, 1.0, 0.3, 0.05,
            help="Lower = more text detected, higher = only confident results"
        )

        st.markdown("### 🧠 Model")

        from src.extractor import get_available_models
        available_models = get_available_models()
        model_names = list(available_models.keys())

        selected_model_name = st.selectbox(
            "Select Model", options=model_names,
            help="EasyOCR works instantly. Others need training."
        )
        selected_model_info = available_models[selected_model_name]

        if selected_model_info['type'] == 'easyocr':
            st.markdown('<span class="badge badge-ready">✅ Ready to use</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-train">🧠 Needs training</span>', unsafe_allow_html=True)

        st.markdown("### 📊 Display")
        show_boxes = st.toggle("Show detection boxes", value=True)
        show_words = st.toggle("Show word details", value=True)

        st.markdown("### 🔌 Backend")
        use_api = st.toggle("Use FastAPI", value=False,
                             help="Connect to FastAPI at :8000")
        if use_api:
            try:
                r = requests.get(f"{BACKEND_URL}/api/health", timeout=2)
                if r.status_code == 200:
                    st.markdown('<span class="badge badge-ready">🟢 Connected</span>', unsafe_allow_html=True)
                else:
                    st.warning("Backend error")
                    use_api = False
            except Exception:
                st.error("Start backend:\n`python backend.py`")
                use_api = False

        # Show comparison results if available
        comparison_path = 'models/comparison_results.json'
        if os.path.exists(comparison_path):
            st.markdown("### 🏆 Accuracy")
            import json
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
            for name, data in comparison.items():
                best_acc = max(d['test_accuracy'] for d in comparison.values())
                emoji = "🥇" if data['test_accuracy'] == best_acc else "📊"
                st.markdown(f"{emoji} **{name}**: {data['test_accuracy']}%")

    # ── Main Content ──────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Left Column: Upload ──
    with col_upload:
        st.markdown('<div class="glass-card-header">📷 Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload handwritten Tamil image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Drag and drop or click to browse",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            # Image info bar
            st.markdown(f"""
            <div class="info-bar">
                <div class="info-item">📐 <span>{image.size[0]}×{image.size[1]}</span></div>
                <div class="info-item">📁 <span>{uploaded_file.size / 1024:.0f} KB</span></div>
                <div class="info-item">🎨 <span>{image.mode}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Drop your image here</div>
                <div class="upload-hint">JPG, PNG, BMP, TIFF • Max 200MB</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right Column: Results ──
    with col_result:
        st.markdown('<div class="glass-card-header">📝 Extracted Text</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            extract_btn = st.button("✨ Extract Text", use_container_width=True, type="primary")

            if extract_btn:
                with st.spinner(""):
                    # Show custom progress
                    progress_placeholder = st.empty()
                    progress_placeholder.markdown("""
                    <div style="text-align:center; padding: 20px;">
                        <div style="font-size: 2rem; animation: float 1s ease-in-out infinite;">🔍</div>
                        <div style="color: #94a3b8; margin-top: 8px;">Analyzing handwriting...</div>
                    </div>
                    """, unsafe_allow_html=True)

                    try:
                        text = ""
                        details = []

                        if use_api:
                            uploaded_file.seek(0)
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            params = {"model": selected_model_info['type'], "confidence_threshold": confidence_threshold}
                            resp = requests.post(f"{BACKEND_URL}/api/extract", files=files, params=params, timeout=60)
                            if resp.status_code == 200:
                                data = resp.json()
                                text = data['text']
                                details = [{'label': r['label'], 'confidence': r['confidence'], 'bbox': tuple(r['bbox'])} for r in data['regions']]
                            else:
                                st.error(f"API Error: {resp.json().get('error', resp.text)}")
                        else:
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
                            text, details = extractor.predict_text(temp_path, confidence_threshold=confidence_threshold)
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                        progress_placeholder.empty()

                        # ── Show Results ──
                        if text:
                            # Extracted text
                            st.markdown(f"""
                            <div class="result-box">
                                <div class="result-text">{text}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Stats row
                            avg_conf = np.mean([d['confidence'] for d in details]) if details else 0
                            high_conf = sum(1 for d in details if d['confidence'] > 0.8)
                            conf_color = "#34d399" if avg_conf > 0.7 else "#fbbf24" if avg_conf > 0.4 else "#f87171"

                            st.markdown(f"""
                            <div class="stats-row">
                                <div class="stat-pill">
                                    <div class="stat-value">{len(details)}</div>
                                    <div class="stat-label">Regions</div>
                                </div>
                                <div class="stat-pill">
                                    <div class="stat-value" style="color: {conf_color}">{avg_conf:.0%}</div>
                                    <div class="stat-label">Confidence</div>
                                </div>
                                <div class="stat-pill">
                                    <div class="stat-value">{high_conf}</div>
                                    <div class="stat-label">High Conf.</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Download button
                            st.download_button(
                                "📥 Download Text",
                                data=text,
                                file_name="tamil_extracted.txt",
                                mime="text/plain",
                                use_container_width=True
                            )

                            # Tabs for extra views
                            if show_words or show_boxes:
                                tabs = []
                                tab_labels = []
                                if show_boxes:
                                    tab_labels.append("🔲 Detection Map")
                                if show_words:
                                    tab_labels.append("🏷️ Word Details")

                                if tab_labels:
                                    created_tabs = st.tabs(tab_labels)
                                    tab_idx = 0

                                    if show_boxes and details:
                                        with created_tabs[tab_idx]:
                                            img_draw = np.array(image.copy())
                                            for d in details:
                                                x, y, w, h = d['bbox']
                                                conf = d['confidence']
                                                if conf > 0.7:
                                                    color = (16, 185, 129)  # green
                                                elif conf > 0.4:
                                                    color = (251, 191, 36)  # yellow
                                                else:
                                                    color = (248, 113, 113)  # red
                                                cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, 2)
                                                # Add label
                                                label = f"{d['label']} ({conf:.0%})"
                                                font_scale = 0.5
                                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                                                cv2.rectangle(img_draw, (x, y - th - 8), (x + tw + 4, y), color, -1)
                                                cv2.putText(img_draw, label, (x + 2, y - 4),
                                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                                            st.image(img_draw, use_container_width=True)
                                        tab_idx += 1

                                    if show_words and details:
                                        with created_tabs[tab_idx]:
                                            # Word tags
                                            tags_html = ""
                                            for d in details:
                                                conf = d['confidence']
                                                conf_class = "conf-high" if conf > 0.7 else "conf-mid" if conf > 0.4 else "conf-low"
                                                tags_html += f"""
                                                <span class="region-tag">
                                                    <span class="region-word">{d['label']}</span>
                                                    <span class="region-conf {conf_class}">{conf:.0%}</span>
                                                </span>"""
                                            st.markdown(tags_html, unsafe_allow_html=True)

                        else:
                            st.markdown("""
                            <div style="text-align:center; padding: 40px; color: #94a3b8;">
                                <div style="font-size: 2.5rem;">🤔</div>
                                <div style="margin-top: 8px; font-weight: 500;">No text detected</div>
                                <div style="color: #475569; font-size: 0.85rem; margin-top: 4px;">
                                    Try a clearer image or lower the confidence threshold
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"Error: {str(e)}")
                        with st.expander("Show details"):
                            st.exception(e)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 20px;">
                <div style="font-size: 3.5rem; margin-bottom: 12px;">📄</div>
                <div style="color: #94a3b8; font-size: 1.05rem; font-weight: 500;">
                    Upload an image to get started
                </div>
                <div style="color: #475569; font-size: 0.85rem; margin-top: 8px;">
                    Supports handwritten Tamil text in JPG, PNG, BMP, TIFF
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        ✍️ Tamil Text Extractor • EasyOCR + CNN + MobileNetV2 •
        <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
