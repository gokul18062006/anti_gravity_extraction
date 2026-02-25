"""
preprocess.py — Data loading, preprocessing, and augmentation for Tamil OCR.

Supports both grayscale (1-channel for custom CNN) and
RGB (3-channel for MobileNetV2 transfer learning) modes.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import json

# Image dimensions for the model
IMG_HEIGHT = 64
IMG_WIDTH = 64


def load_dataset(data_dir, color_mode='grayscale'):
    """
    Load images and labels from directory structure:
      data_dir/<class_label>/image_files

    Args:
        data_dir: Path to the dataset directory
        color_mode: 'grayscale' (1 channel) or 'rgb' (3 channels)

    Returns:
        images: numpy array of shape (N, H, W, C) where C=1 or 3
        labels: numpy array of integer labels
        class_names: list of class folder names (sorted)
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    # Filter out non-directories
    class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]

    print(f"Found {len(class_names)} classes in '{data_dir}'")
    print(f"Color mode: {color_mode}")

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        file_list = os.listdir(class_dir)
        for img_file in file_list:
            img_path = os.path.join(class_dir, img_file)
            img = load_single_image(img_path, color_mode)
            if img is None:
                continue
            images.append(img)
            labels.append(label_idx)

        if (label_idx + 1) % 20 == 0:
            print(f"  Loaded {label_idx + 1}/{len(class_names)} classes...")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"Total images loaded: {len(images)}")
    print(f"Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
    return images, labels, class_names


def load_single_image(img_path, color_mode='grayscale'):
    """
    Load and preprocess a single image.

    Args:
        img_path: Path to the image file
        color_mode: 'grayscale' or 'rgb'

    Returns:
        Preprocessed image array or None if loading fails
    """
    if color_mode == 'rgb':
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image_rgb(img)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = preprocess_image(img)
    return img


def preprocess_image(img):
    """
    Preprocess a single grayscale image (for custom CNN):
    1. Resize to IMG_HEIGHT x IMG_WIDTH
    2. Normalize pixel values to [0, 1]
    3. Reshape to (H, W, 1)
    """
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
    return img


def preprocess_image_rgb(img):
    """
    Preprocess a single RGB image (for MobileNetV2):
    1. Resize to IMG_HEIGHT x IMG_WIDTH
    2. Normalize pixel values to [0, 1]
    3. Shape remains (H, W, 3)
    """
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img


def grayscale_to_rgb(images):
    """
    Convert grayscale images (H, W, 1) to RGB (H, W, 3) by repeating channels.
    Useful when you have grayscale data but need RGB for MobileNet.
    """
    if images.shape[-1] == 1:
        return np.repeat(images, 3, axis=-1)
    return images


def split_dataset(images, labels, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    Default: 80% train, 10% val, 10% test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_ratio, random_state=random_state, stratify=labels
    )
    # Second split: separate validation from training
    val_fraction = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_augmentation_generator():
    """
    Create an ImageDataGenerator with augmentation for training.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )


def create_label_mapping(class_names):
    """
    Create a mapping from class index to Tamil character label.

    Returns:
        dict: {index (str): class_name}
    """
    mapping = {str(i): name for i, name in enumerate(class_names)}
    return mapping


def save_label_mapping(mapping, filepath):
    """Save label mapping to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Label mapping saved to '{filepath}'")


def load_label_mapping(filepath):
    """Load label mapping from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping
