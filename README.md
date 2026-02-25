# 📝 AI-Powered Tamil Handwritten Text Extraction

An AI-powered application that extracts Tamil text from handwritten images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, with a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

---

## 🏗️ Architecture

```
Input Image → Preprocessing → Character Segmentation → CNN Classification → Tamil Text Output
```

| Component | Technology | Description |
|-----------|-----------|-------------|
| Preprocessing | OpenCV | Grayscale, binarization, noise removal |
| Segmentation | OpenCV Contours | Isolate individual characters |
| Recognition | TensorFlow CNN | 4-layer CNN with BatchNorm |
| Web UI | Streamlit | Image upload & result display |

---

## 📁 Project Structure

```
text_extraction/
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── README.md             # Dataset download instructions
│   ├── train/                # Training images (by class)
│   └── test/                 # Test images (by class)
├── models/
│   ├── tamil_ocr_model.h5    # Trained model (generated)
│   └── label_mapping.json    # Character labels (generated)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Data loading & preprocessing
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training script
│   └── extractor.py          # Text extraction pipeline
└── samples/                  # Sample test images
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Tamil Handwritten Character Recognition dataset](https://www.kaggle.com/datasets/sudalairajkumar/tamil-handwritten-character-recognition) from Kaggle.

Extract and organize into:
```
data/train/<class_label>/image_files
data/test/<class_label>/image_files
```

See `data/README.md` for detailed instructions.

### 3. Train the Model

```bash
# Full training (recommended)
python -m src.train --epochs 50 --batch_size 32

# Quick test run (2 epochs, 100 samples per class)
python -m src.train --epochs 2 --subset 100
```

### 4. Run the Web App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and upload a handwritten Tamil image!

---

## 🧠 Model Architecture

```
Input (64×64×1)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Conv2D(256) + BatchNorm + MaxPool
  → Dense(512) + Dropout(0.5)
  → Dense(256) + Dropout(0.3)
  → Dense(num_classes) + Softmax
```

**Training Features:**
- Data augmentation (rotation, shift, zoom, shear)
- Early stopping with patience=10
- Learning rate reduction on plateau
- Model checkpointing (saves best model)

---

## 📊 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/train` | Path to training data |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--model_dir` | `models` | Model save directory |
| `--subset` | `None` | Samples per class (for testing) |

---

## 🔧 How It Works

1. **Image Upload** — User uploads a handwritten Tamil text image
2. **Preprocessing** — Convert to grayscale, apply adaptive thresholding
3. **Segmentation** — Detect character contours, sort in reading order
4. **Classification** — Feed each character into the CNN model
5. **Assembly** — Combine predicted characters into Tamil text
6. **Display** — Show results with confidence scores and visualization

---

## 📋 Requirements

- Python 3.9 or higher
- GPU recommended for training (CPU works but is slower)
- ~500 MB disk space for dataset
- ~100 MB for trained model

---

## 👨‍💻 Author

College Mini Project — AI-Powered Tamil Handwritten Text Extraction

---

## 📄 License

This project is for educational purposes.
