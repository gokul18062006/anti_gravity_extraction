"""
extractor.py — Tamil handwritten text extraction from images.

Supports three extraction models:
  1. EasyOCR (PRIMARY)   — Pre-trained, works immediately, no dataset needed
  2. Custom CNN          — Requires dataset + training
  3. MobileNetV2         — Requires dataset + training (transfer learning)
"""

import os
import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 1: EasyOCR (PRIMARY — No training required)
# ═══════════════════════════════════════════════════════════════════════

class EasyOCRExtractor:
    """
    Tamil text extraction using EasyOCR.
    Works out of the box — no dataset download or training needed.
    """

    def __init__(self):
        """Initialize EasyOCR reader with Tamil support."""
        import easyocr
        import shutil
        print("Loading EasyOCR model (first time may download ~100MB)...")

        # Try multiple initialization strategies to avoid model mismatch
        strategies = [
            {'langs': ['ta'], 'desc': 'Tamil only'},
            {'langs': ['ta', 'en'], 'desc': 'Tamil + English'},
            {'langs': ['en'], 'desc': 'English fallback'},
        ]

        for strategy in strategies:
            try:
                self.reader = easyocr.Reader(
                    strategy['langs'],
                    gpu=False,
                    verbose=False
                )
                self.languages = strategy['langs']
                print(f"EasyOCR ready! ({strategy['desc']})")
                return
            except RuntimeError as e:
                print(f"Strategy '{strategy['desc']}' failed: {e}")
                # Clear corrupted cache and retry
                cache_dir = os.path.join(os.path.expanduser('~'), '.EasyOCR')
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print("Cleared EasyOCR cache, retrying...")
                    try:
                        self.reader = easyocr.Reader(
                            strategy['langs'],
                            gpu=False,
                            verbose=False
                        )
                        self.languages = strategy['langs']
                        print(f"EasyOCR ready after cache clear! ({strategy['desc']})")
                        return
                    except Exception:
                        continue

        raise RuntimeError(
            "Could not initialize EasyOCR. Please try: "
            "pip install --upgrade easyocr torch torchvision"
        )

    def predict_text(self, image_path, confidence_threshold=0.3):
        """
        Extract Tamil text from a handwritten image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence to include text

        Returns:
            text: Extracted Tamil text string
            details: List of dicts with detection info
        """
        results = self.reader.readtext(image_path)

        text_parts = []
        details = []

        for (bbox, text, confidence) in results:
            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            x_coords = [int(p[0]) for p in bbox]
            y_coords = [int(p[1]) for p in bbox]
            x, y = min(x_coords), min(y_coords)
            w, h = max(x_coords) - x, max(y_coords) - y

            detail = {
                'label': text,
                'confidence': float(confidence),
                'bbox': (x, y, w, h)
            }
            details.append(detail)

            if confidence >= confidence_threshold:
                text_parts.append(text)

        full_text = ' '.join(text_parts)
        return full_text, details

    def visualize_segmentation(self, image_path, output_path=None):
        """Draw bounding boxes on detected text regions."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: '{image_path}'")

        _, details = self.predict_text(image_path)

        for d in details:
            x, y, w, h = d['bbox']
            color = (0, 255, 0) if d['confidence'] > 0.5 else (0, 165, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 2 & 3: CNN / MobileNetV2 (Requires training)
# ═══════════════════════════════════════════════════════════════════════

class CNNExtractor:
    """
    Tamil text extraction using trained CNN or MobileNetV2 model.
    Requires training first via: python -m src.train
    """

    def __init__(self, model_path, mapping_path='models/label_mapping.json'):
        """Load trained model and label mapping."""
        from tensorflow.keras.models import load_model
        from src.preprocess import load_label_mapping, IMG_HEIGHT, IMG_WIDTH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Train first: python -m src.train"
            )

        self.model = load_model(model_path)
        self.label_mapping = load_label_mapping(mapping_path)
        self.img_h = IMG_HEIGHT
        self.img_w = IMG_WIDTH

        # Auto-detect input channels
        self.input_channels = self.model.input_shape[-1]
        self.model_type = 'MobileNetV2' if self.input_channels == 3 else 'Custom CNN'

        print(f"Loaded {self.model_type} from '{model_path}'")
        print(f"Classes: {len(self.label_mapping)}")

    def predict_text(self, image_path, confidence_threshold=0.3):
        """Extract text using character segmentation + CNN classification."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: '{image_path}'")

        char_images, bounding_boxes = self._segment_characters(image)
        if not char_images:
            return "", []

        text_parts = []
        details = []

        for char_img, bbox in zip(char_images, bounding_boxes):
            label, confidence = self._predict_character(char_img)
            details.append({
                'label': label,
                'confidence': confidence,
                'bbox': bbox
            })
            text_parts.append(label if confidence >= confidence_threshold else '?')

        return ''.join(text_parts), details

    def _segment_characters(self, image):
        """Segment individual characters using contour detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=11, C=5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = image.shape[0] * image.shape[1]

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < img_area * 0.001 or area > img_area * 0.5 or w < 5 or h < 5:
                continue
            boxes.append((x, y, w, h))

        # Sort: top-to-bottom, left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1] // 30, b[0]))

        char_images = [gray[y:y+h, x:x+w] for x, y, w, h in boxes]
        return char_images, boxes

    def _predict_character(self, char_img):
        """Classify a single character image."""
        resized = cv2.resize(char_img, (self.img_w, self.img_h))
        if np.mean(resized) > 127:
            resized = 255 - resized
        normalized = resized.astype(np.float32) / 255.0

        if self.input_channels == 3:
            processed = np.stack([normalized] * 3, axis=-1).reshape(1, self.img_h, self.img_w, 3)
        else:
            processed = normalized.reshape(1, self.img_h, self.img_w, 1)

        preds = self.model.predict(processed, verbose=0)
        idx = np.argmax(preds[0])
        return self.label_mapping.get(str(idx), '?'), float(preds[0][idx])


# ═══════════════════════════════════════════════════════════════════════
#  Utility: discover available models
# ═══════════════════════════════════════════════════════════════════════

def get_available_models(model_dir='models'):
    """
    List all available models for the UI.

    Returns:
        dict: {display_name: {'type': 'easyocr'|'cnn'|'mobilenet', 'path': ...}}
    """
    models = {
        'EasyOCR (Pre-trained)': {'type': 'easyocr', 'path': None}
    }

    cnn_path = os.path.join(model_dir, 'tamil_ocr_model_cnn.h5')
    mobilenet_path = os.path.join(model_dir, 'tamil_ocr_model_mobilenet.h5')

    if os.path.exists(cnn_path):
        models['Custom CNN (Trained)'] = {'type': 'cnn', 'path': cnn_path}
    if os.path.exists(mobilenet_path):
        models['MobileNetV2 (Transfer Learning)'] = {'type': 'mobilenet', 'path': mobilenet_path}

    return models


def load_extractor(model_info):
    """
    Load the appropriate extractor based on model info.

    Args:
        model_info: dict with 'type' and 'path' keys

    Returns:
        Extractor instance (EasyOCRExtractor or CNNExtractor)
    """
    if model_info['type'] == 'easyocr':
        return EasyOCRExtractor()
    else:
        return CNNExtractor(model_info['path'])
