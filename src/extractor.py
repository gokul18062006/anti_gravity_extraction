"""
extractor.py — Character segmentation and text extraction from handwritten Tamil images.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import IMG_HEIGHT, IMG_WIDTH, load_label_mapping


class TamilTextExtractor:
    """
    Extract Tamil text from handwritten images using:
    1. Image preprocessing (binarization, noise removal)
    2. Character segmentation (contour detection)
    3. CNN/MobileNetV2 character classification
    4. Text assembly

    Automatically detects model input channels (1=CNN, 3=MobileNet).
    """

    def __init__(self, model_path='models/tamil_ocr_model_cnn.h5',
                 mapping_path='models/label_mapping.json'):
        """
        Initialize the extractor with a trained model and label mapping.
        Auto-detects if model expects grayscale (1ch) or RGB (3ch) input.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Please train the model first using: python -m src.train"
            )
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Label mapping not found at '{mapping_path}'. "
                "Please train the model first using: python -m src.train"
            )

        self.model = load_model(model_path)
        self.label_mapping = load_label_mapping(mapping_path)

        # Auto-detect input channels from model
        input_shape = self.model.input_shape  # e.g. (None, 64, 64, 1) or (None, 64, 64, 3)
        self.input_channels = input_shape[-1]
        self.model_type = 'mobilenet' if self.input_channels == 3 else 'cnn'

        print(f"Loaded model from '{model_path}'")
        print(f"Model type: {self.model_type} (input channels: {self.input_channels})")
        print(f"Loaded {len(self.label_mapping)} character classes")

    def preprocess_for_segmentation(self, image):
        """
        Preprocess a full handwritten image for character segmentation.

        Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Apply adaptive thresholding for binarization
        4. Apply morphological operations to clean up

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            binary: Binary image ready for contour detection
            gray: Grayscale version of the input
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=5
        )

        # Morphological cleaning: close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary, gray

    def segment_characters(self, image):
        """
        Segment individual characters from a handwritten text image.

        Uses contour detection and sorts bounding boxes
        left-to-right (with basic line detection for multi-line text).

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            char_images: List of cropped character images (grayscale)
            bounding_boxes: List of (x, y, w, h) tuples
        """
        binary, gray = self.preprocess_for_segmentation(image)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and extract bounding boxes
        bounding_boxes = []
        img_area = image.shape[0] * image.shape[1]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter out very small or very large contours
            if area < img_area * 0.001:  # Too small (noise)
                continue
            if area > img_area * 0.5:  # Too large (entire image)
                continue
            if w < 5 or h < 5:  # Minimum size
                continue

            bounding_boxes.append((x, y, w, h))

        # Sort bounding boxes: top-to-bottom, then left-to-right
        # Group by lines (boxes with similar y-coordinates)
        bounding_boxes = self._sort_bounding_boxes(bounding_boxes)

        # Extract character images
        char_images = []
        for (x, y, w, h) in bounding_boxes:
            char_img = gray[y:y+h, x:x+w]
            char_images.append(char_img)

        return char_images, bounding_boxes

    def _sort_bounding_boxes(self, boxes):
        """
        Sort bounding boxes in reading order:
        top-to-bottom by line, left-to-right within each line.
        """
        if not boxes:
            return boxes

        # Sort by y-coordinate first
        boxes = sorted(boxes, key=lambda b: b[1])

        # Group into lines based on y-overlap
        lines = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            # Check if this box is on the same line as the current line
            prev_y = current_line[-1][1]
            prev_h = current_line[-1][3]
            curr_y = box[1]

            # If vertical overlap > 50%, same line
            if abs(curr_y - prev_y) < prev_h * 0.5:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

        lines.append(current_line)

        # Sort each line left-to-right and flatten
        sorted_boxes = []
        for line in lines:
            line_sorted = sorted(line, key=lambda b: b[0])
            sorted_boxes.extend(line_sorted)

        return sorted_boxes

    def preprocess_for_prediction(self, char_img):
        """
        Preprocess a single character image for model prediction.
        Handles both CNN (1-channel) and MobileNet (3-channel) models.

        Args:
            char_img: Grayscale character image (any size)

        Returns:
            processed: Preprocessed image ready for model (1, 64, 64, C)
        """
        # Resize to model input size
        resized = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))

        # Invert if background is white (model expects white text on black bg)
        if np.mean(resized) > 127:
            resized = 255 - resized

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        if self.input_channels == 3:
            # MobileNetV2: convert grayscale to 3-channel
            normalized = np.stack([normalized] * 3, axis=-1)
            processed = normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
        else:
            # Custom CNN: 1-channel grayscale
            processed = normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

        return processed

    def predict_character(self, char_img):
        """
        Predict a single character.

        Args:
            char_img: Grayscale character image

        Returns:
            predicted_label: The predicted Tamil character label
            confidence: Prediction confidence (0-1)
        """
        processed = self.preprocess_for_prediction(char_img)
        predictions = self.model.predict(processed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = self.label_mapping.get(str(predicted_class), '?')
        return predicted_label, confidence

    def predict_text(self, image_path, confidence_threshold=0.3):
        """
        Extract Tamil text from a handwritten image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence to include a character

        Returns:
            text: Extracted Tamil text string
            details: List of dicts with char info (label, confidence, bbox)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: '{image_path}'")

        # Segment characters
        char_images, bounding_boxes = self.segment_characters(image)

        if not char_images:
            return "", []

        # Predict each character
        text_parts = []
        details = []

        for char_img, bbox in zip(char_images, bounding_boxes):
            label, confidence = self.predict_character(char_img)

            detail = {
                'label': label,
                'confidence': confidence,
                'bbox': bbox
            }
            details.append(detail)

            if confidence >= confidence_threshold:
                text_parts.append(label)
            else:
                text_parts.append('?')  # Low confidence placeholder

        text = ''.join(text_parts)
        return text, details

    def visualize_segmentation(self, image_path, output_path=None):
        """
        Visualize character segmentation with bounding boxes on the image.

        Args:
            image_path: Path to the input image
            output_path: Path to save the visualization (optional)

        Returns:
            annotated_image: Image with bounding boxes drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: '{image_path}'")

        char_images, bounding_boxes = self.segment_characters(image)

        annotated = image.copy()
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, str(i), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Segmentation visualization saved to '{output_path}'")

        return annotated


def predict_text(image_path,
                 model_path='models/tamil_ocr_model_cnn.h5',
                 mapping_path='models/label_mapping.json'):
    """
    Convenience function for quick text extraction.

    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        mapping_path: Path to the label mapping JSON

    Returns:
        Extracted Tamil text string
    """
    extractor = TamilTextExtractor(model_path, mapping_path)
    text, _ = extractor.predict_text(image_path)
    return text


def get_available_models(model_dir='models'):
    """
    List available trained models.

    Returns:
        dict: {display_name: model_path}
    """
    models = {}
    cnn_path = os.path.join(model_dir, 'tamil_ocr_model_cnn.h5')
    mobilenet_path = os.path.join(model_dir, 'tamil_ocr_model_mobilenet.h5')

    if os.path.exists(cnn_path):
        models['Custom CNN'] = cnn_path
    if os.path.exists(mobilenet_path):
        models['MobileNetV2 (Transfer Learning)'] = mobilenet_path

    return models
