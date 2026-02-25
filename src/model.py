"""
model.py — CNN & MobileNetV2 architectures for Tamil handwritten character recognition.

Supports two model types:
  1. 'cnn'       — Custom CNN built from scratch
  2. 'mobilenet' — MobileNetV2 with transfer learning (pre-trained on ImageNet)
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 1: Custom CNN (built from scratch)
# ═══════════════════════════════════════════════════════════════════════

def build_cnn_model(num_classes, input_shape=(64, 64, 1)):
    """
    Build a custom CNN model for Tamil character classification.

    Architecture:
        Conv2D(32) → BatchNorm → MaxPool
        Conv2D(64) → BatchNorm → MaxPool
        Conv2D(128) → BatchNorm → MaxPool
        Conv2D(256) → BatchNorm → MaxPool
        Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Softmax

    Args:
        num_classes: Number of Tamil character classes
        input_shape: Shape of input images (H, W, C)

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Block 1
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 2: MobileNetV2 with Transfer Learning
# ═══════════════════════════════════════════════════════════════════════

def build_mobilenet_model(num_classes, input_shape=(64, 64, 3)):
    """
    Build a MobileNetV2 transfer learning model for Tamil character classification.

    Uses MobileNetV2 pre-trained on ImageNet as feature extractor,
    with a custom classification head for Tamil characters.

    Architecture:
        MobileNetV2 (frozen base) → GlobalAvgPool → Dense(512) → Dropout
        → Dense(256) → Dropout → Dense(num_classes) → Softmax

    Args:
        num_classes: Number of Tamil character classes
        input_shape: Shape of input images (H, W, 3) — must be 3 channels

    Returns:
        Compiled Keras model
    """
    # Load MobileNetV2 pre-trained on ImageNet (without top classifier)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model (don't train pre-trained weights initially)
    base_model.trainable = False

    # Build custom classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def unfreeze_mobilenet(model, num_layers_to_unfreeze=20):
    """
    Fine-tune the MobileNetV2 model by unfreezing the last N layers.
    Call this after initial training to improve accuracy further.

    Args:
        model: The MobileNetV2 model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    """
    # The base model is the second layer (index 1) in our Model
    base_model = model.layers[1]
    base_model.trainable = True

    # Freeze all layers except the last N
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Re-compile with a lower learning rate for fine-tuning
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable = sum(1 for layer in base_model.layers if layer.trainable)
    frozen = sum(1 for layer in base_model.layers if not layer.trainable)
    print(f"Fine-tuning: {trainable} trainable layers, {frozen} frozen layers")

    return model


# ═══════════════════════════════════════════════════════════════════════
#  Unified build function
# ═══════════════════════════════════════════════════════════════════════

def build_model(num_classes, model_type='cnn', input_shape=None):
    """
    Build a model based on the specified type.

    Args:
        num_classes: Number of Tamil character classes
        model_type: 'cnn' for custom CNN, 'mobilenet' for MobileNetV2
        input_shape: Override default input shape (optional)

    Returns:
        Compiled Keras model
    """
    if model_type == 'cnn':
        shape = input_shape or (64, 64, 1)
        print(f"Building Custom CNN model with input shape {shape}")
        return build_cnn_model(num_classes, shape)
    elif model_type == 'mobilenet':
        shape = input_shape or (64, 64, 3)
        print(f"Building MobileNetV2 model with input shape {shape}")
        return build_mobilenet_model(num_classes, shape)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Use 'cnn' or 'mobilenet'.")


def get_callbacks(model_save_path='models/tamil_ocr_model.h5'):
    """
    Get training callbacks for model optimization.

    Returns:
        List of Keras callbacks:
        - EarlyStopping: stop if val_loss doesn't improve for 10 epochs
        - ModelCheckpoint: save the best model
        - ReduceLROnPlateau: reduce LR if val_loss plateaus
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def print_model_summary(model):
    """Print a summary of the model architecture."""
    model.summary()
    total_params = model.count_params()
    trainable_params = sum(
        int(w.numpy().size) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params
    print(f"\nTotal parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
