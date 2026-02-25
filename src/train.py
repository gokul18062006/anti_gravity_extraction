"""
train.py — Training script for the Tamil OCR models.

Supports two model types:
  - 'cnn'       — Custom CNN (grayscale input)
  - 'mobilenet' — MobileNetV2 transfer learning (RGB input)
  - 'both'      — Train both models and compare results

Usage:
    python -m src.train --model_type cnn
    python -m src.train --model_type mobilenet
    python -m src.train --model_type both          # Train & compare both
    python -m src.train --epochs 2 --subset 100    # Quick smoke test
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    load_dataset, split_dataset, get_data_augmentation_generator,
    create_label_mapping, save_label_mapping, grayscale_to_rgb,
    IMG_HEIGHT, IMG_WIDTH
)
from src.model import build_model, get_callbacks, print_model_summary, unfreeze_mobilenet


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tamil OCR Model')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only N samples per class (for testing)')
    parser.add_argument('--model_type', type=str, default='both',
                        choices=['cnn', 'mobilenet', 'both'],
                        help="Model type: 'cnn', 'mobilenet', or 'both' (default: both)")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune MobileNetV2 after initial training')
    return parser.parse_args()


def plot_training_history(history, save_path, title=''):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to '{save_path}'")


def plot_comparison(results, save_path):
    """
    Plot a comparison bar chart between CNN and MobileNetV2 results.
    """
    models = list(results.keys())
    accuracies = [results[m]['test_accuracy'] * 100 for m in models]
    train_times = [results[m].get('epochs_trained', 0) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Comparison: Custom CNN vs MobileNetV2', fontsize=16, fontweight='bold')

    # Accuracy comparison
    colors = ['#667eea', '#f093fb']
    bars1 = axes[0].bar(models, accuracies, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Test Accuracy (%)', fontsize=14)
    axes[0].set_ylim(0, 105)
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    # Epochs trained comparison
    bars2 = axes[1].bar(models, train_times, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_title('Epochs Trained', fontsize=14)
    for bar, ep in zip(bars2, train_times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(ep), ha='center', fontweight='bold', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to '{save_path}'")


def train_single_model(model_type, X_train, X_val, X_test,
                        y_train_cat, y_val_cat, y_test_cat,
                        num_classes, args):
    """
    Train a single model (either CNN or MobileNetV2).

    Returns:
        dict with results (accuracy, loss, epochs, model path, etc.)
    """
    model_name = 'Custom CNN' if model_type == 'cnn' else 'MobileNetV2'
    print("\n" + "🔷" * 30)
    print(f"  TRAINING: {model_name}")
    print("🔷" * 30)

    # Prepare data based on model type
    if model_type == 'mobilenet':
        # MobileNetV2 needs 3-channel (RGB) input
        X_tr = grayscale_to_rgb(X_train) if X_train.shape[-1] == 1 else X_train
        X_v = grayscale_to_rgb(X_val) if X_val.shape[-1] == 1 else X_val
        X_te = grayscale_to_rgb(X_test) if X_test.shape[-1] == 1 else X_test
    else:
        X_tr, X_v, X_te = X_train, X_val, X_test

    # ── Build Model ───────────────────────────────────────────────
    print(f"\nBuilding {model_name}...")
    model = build_model(num_classes, model_type=model_type)
    print_model_summary(model)

    # ── Data Augmentation ─────────────────────────────────────────
    datagen = get_data_augmentation_generator()
    datagen.fit(X_tr)

    # ── Train ─────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    suffix = '_cnn' if model_type == 'cnn' else '_mobilenet'
    model_save_path = os.path.join(args.model_dir, f'tamil_ocr_model{suffix}.h5')
    callbacks = get_callbacks(model_save_path)

    print(f"\nTraining {model_name} for {args.epochs} epochs...")
    history = model.fit(
        datagen.flow(X_tr, y_train_cat, batch_size=args.batch_size),
        validation_data=(X_v, y_val_cat),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── Fine-tune MobileNetV2 (optional) ──────────────────────────
    fine_tune_history = None
    if model_type == 'mobilenet' and args.fine_tune:
        print(f"\n{'='*60}")
        print("FINE-TUNING: Unfreezing last 20 MobileNetV2 layers...")
        print(f"{'='*60}")

        model = unfreeze_mobilenet(model, num_layers_to_unfreeze=20)
        fine_tune_epochs = max(args.epochs // 2, 10)
        fine_tune_callbacks = get_callbacks(model_save_path)

        fine_tune_history = model.fit(
            datagen.flow(X_tr, y_train_cat, batch_size=args.batch_size),
            validation_data=(X_v, y_val_cat),
            epochs=fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            verbose=1
        )

    # ── Evaluate ──────────────────────────────────────────────────
    test_loss, test_accuracy = model.evaluate(X_te, y_test_cat, verbose=0)
    print(f"\n{model_name} Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Save training history plot
    history_path = os.path.join(args.model_dir, f'training_history{suffix}.png')
    plot_training_history(history, history_path, title=f'{model_name} Training')

    # Count actual epochs trained
    epochs_trained = len(history.history['accuracy'])
    if fine_tune_history:
        epochs_trained += len(fine_tune_history.history['accuracy'])

    return {
        'model_name': model_name,
        'model_type': model_type,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'epochs_trained': epochs_trained,
        'model_path': model_save_path,
        'history_path': history_path,
        'total_params': model.count_params()
    }


def main():
    args = parse_args()

    # ── Step 1: Load Dataset ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading dataset...")
    print("=" * 60)

    if not os.path.exists(args.data_dir):
        print(f"\nERROR: Data directory '{args.data_dir}' not found!")
        print("Please download the dataset first. See data/README.md for instructions.")
        sys.exit(1)

    # Load as grayscale (we convert to RGB for MobileNet as needed)
    images, labels, class_names = load_dataset(args.data_dir, color_mode='grayscale')
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Optional: use subset for quick testing
    if args.subset:
        print(f"\nUsing subset: {args.subset} samples per class")
        subset_mask = []
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            n = min(args.subset, len(class_indices))
            subset_mask.extend(class_indices[:n])
        subset_mask = np.array(subset_mask)
        images = images[subset_mask]
        labels = labels[subset_mask]
        print(f"Subset total: {len(images)} images")

    # ── Step 2: Split Dataset ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Splitting dataset...")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ── Step 3: Save label mapping ────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    label_mapping = create_label_mapping(class_names)
    mapping_path = os.path.join(args.model_dir, 'label_mapping.json')
    save_label_mapping(label_mapping, mapping_path)

    # ── Step 4: Train model(s) ────────────────────────────────────
    models_to_train = []
    if args.model_type == 'both':
        models_to_train = ['cnn', 'mobilenet']
    else:
        models_to_train = [args.model_type]

    results = {}
    for mt in models_to_train:
        result = train_single_model(
            mt, X_train, X_val, X_test,
            y_train_cat, y_val_cat, y_test_cat,
            num_classes, args
        )
        results[result['model_name']] = result

    # ── Step 5: Comparison (if both models trained) ───────────────
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"\n{'Model':<20} {'Accuracy':>10} {'Loss':>10} {'Params':>15} {'Epochs':>8}")
        print("-" * 65)
        for name, r in results.items():
            print(f"{name:<20} {r['test_accuracy']*100:>9.2f}% {r['test_loss']:>10.4f} "
                  f"{r['total_params']:>15,} {r['epochs_trained']:>8}")

        # Save comparison plot
        comparison_path = os.path.join(args.model_dir, 'model_comparison.png')
        plot_comparison(results, comparison_path)

        # Save comparison results to JSON
        comparison_json_path = os.path.join(args.model_dir, 'comparison_results.json')
        comparison_data = {}
        for name, r in results.items():
            comparison_data[name] = {
                'test_accuracy': round(r['test_accuracy'] * 100, 2),
                'test_loss': round(r['test_loss'], 4),
                'total_params': r['total_params'],
                'epochs_trained': r['epochs_trained'],
                'model_path': r['model_path']
            }
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison results saved to '{comparison_json_path}'")

        # Determine winner
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 WINNER: {best_model[0]} with {best_model[1]['test_accuracy']*100:.2f}% accuracy!")

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  📦 {name}:")
        print(f"     Model:    {r['model_path']}")
        print(f"     Accuracy: {r['test_accuracy']*100:.2f}%")
        print(f"     Plot:     {r['history_path']}")
    print(f"\n  📝 Label mapping: {mapping_path}")


if __name__ == '__main__':
    main()
