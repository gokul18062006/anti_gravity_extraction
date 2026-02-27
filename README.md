# 📝 AI-Powered Tamil Handwritten Text Extraction

Extract Tamil text from handwritten images using AI — powered by **EasyOCR** (pre-trained) with optional **Custom CNN** and **MobileNetV2** models.

---

## 🏗️ Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│   Frontend (Streamlit)      │      │   Backend (FastAPI)          │
│   app.py — port 8501        │─────▶│   backend.py — port 8000     │
│                             │ HTTP │                              │
│  • Upload image             │      │  POST /api/extract           │
│  • Select model             │      │  GET  /api/models            │
│  • View results             │      │  GET  /api/health            │
│  • Download text            │      │  GET  /docs (Swagger UI)     │
└─────────────────────────────┘      └──────────────────────────────┘
                                              │
                                     ┌────────┴────────┐
                                     │   src/ (Core)    │
                                     │  extractor.py    │
                                     │  model.py        │
                                     │  preprocess.py   │
                                     │  train.py        │
                                     └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd c:\Users\gokulp\Desktop\text_extraction
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Backend (FastAPI)
```bash
python backend.py
```
API runs at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

### 3. Run Frontend (Streamlit)
```bash
streamlit run app.py
```
App runs at `http://localhost:8501`

### 4. Upload & Extract
- Open `http://localhost:8501`
- Enable **"Use FastAPI Backend"** in the sidebar (or use direct mode)
- Upload a Tamil handwritten image → click **Extract Text**

---

## 🧠 Available Models

| Model | Type | Setup Required | Accuracy |
|-------|------|---------------|----------|
| **EasyOCR** | Pre-trained | ✅ None | Good |
| **Custom CNN** | Train from scratch | Dataset + training | ~85-90% |
| **MobileNetV2** | Transfer learning | Dataset + training | ~95%+ |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/extract` | Extract text from uploaded image |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/health` | Health check |

### Example: Extract Text via API
```bash
curl -X POST "http://localhost:8000/api/extract?model=easyocr&confidence_threshold=0.3" \
  -F "file=@my_image.png"
```

---

## 📁 Project Structure

```
text_extraction/
├── app.py              ← Streamlit frontend
├── backend.py          ← FastAPI backend
├── requirements.txt    ← Dependencies
├── README.md           ← This file
├── models/             ← Trained models (generated)
└── src/
    ├── extractor.py    ← EasyOCR + CNN + MobileNet extractors
    ├── model.py        ← CNN & MobileNetV2 architectures
    ├── train.py        ← Training script (optional)
    └── preprocess.py   ← Data preprocessing
```

---

## 👨‍💻 Author

College Mini Project — AI-Powered Tamil Handwritten Text Extraction
