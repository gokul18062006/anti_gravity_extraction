# 📝 AI-Powered Tamil Handwritten Text Extraction

Extract Tamil text from handwritten images using AI — powered by **EasyOCR** (pre-trained) with optional **Custom CNN** and **MobileNetV2** models for comparison.

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd c:\Users\gokulp\Desktop\text_extraction
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
streamlit run app.py
```

### 3. Upload & Extract
Open `http://localhost:8501`, upload a Tamil handwritten image, and click **Extract Text**!

> **That's it!** EasyOCR works immediately — no dataset or training needed.

---

## 🧠 Available Models

| Model | Type | Setup Required | Expected Accuracy |
|-------|------|---------------|-------------------|
| **EasyOCR** | Pre-trained | ✅ None — works instantly | Good |
| **Custom CNN** | Train from scratch | Dataset + training | ~85-90% |
| **MobileNetV2** | Transfer learning | Dataset + training | ~95%+ |

### Optional: Train CNN & MobileNet (for comparison)
```bash
# Download dataset from Kaggle, then:
python -m src.train --model_type both --epochs 50
```

---

## 📁 Project Structure

```
text_extraction/
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Trained models (generated)
└── src/
    ├── __init__.py
    ├── extractor.py          # EasyOCR + CNN + MobileNet extractors
    ├── preprocess.py         # Data loading & preprocessing
    ├── model.py              # CNN & MobileNetV2 architectures
    └── train.py              # Training script (optional)
```

---

## 🔧 How It Works

```
Input Image → Preprocessing → Text Detection → Character Recognition → Tamil Text
```

1. **EasyOCR** uses a pre-trained CRAFT detector + CRNN recognizer
2. **Custom CNN** uses contour-based segmentation + 4-layer CNN classifier
3. **MobileNetV2** uses contour segmentation + ImageNet transfer learning

---

## 👨‍💻 Author

College Mini Project — AI-Powered Tamil Handwritten Text Extraction
