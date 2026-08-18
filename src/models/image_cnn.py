"""Supervised image models: a small custom CNN built for single-channel input,
and a transfer-learning path over ResNet50/EfficientNetB0/VGG16 (all three
expect 3-channel input, so we replicate the grayscale channel rather than
retraining a stem from scratch — not worth the extra params for this dataset size).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_BACKBONES = {
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
}


def build_custom_cnn(img_size: int = 224, channels: int = 1) -> models.Model:
    """A from-scratch CNN sized for a few thousand training patches, not
    ImageNet — four conv blocks is already plenty before we start overfitting.
    """
    inputs = layers.Input(shape=(img_size, img_size, channels))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
    return model


class ReplicateChannels(layers.Layer):
    """Grayscale -> 3ch inside the graph, so preprocessing stays 1-channel
    everywhere else and only the model itself knows it needs 3.
    """
    def call(self, x):
        return tf.repeat(x, repeats=3, axis=-1)


def build_transfer_model(backbone_name: str, img_size: int = 224, fine_tune_last_n: int = 20) -> models.Model:
    if backbone_name not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone_name}, choose from {list(_BACKBONES)}")
    backbone_cls, preprocess_fn = _BACKBONES[backbone_name]

    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = ReplicateChannels()(inputs)
    x = layers.Lambda(lambda t: preprocess_fn(t * 255.0))(x)

    base = backbone_cls(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False
    if fine_tune_last_n:
        for layer in base.layers[-fine_tune_last_n:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"transfer_{backbone_name}")
    model.compile(
        optimizer=optimizers.Adam(1e-4),  # lower LR than the custom CNN — we're fine-tuning pretrained weights
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def default_callbacks(checkpoint_path: Path, patience: int = 6) -> list:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_recall", mode="max", save_best_only=True),
    ]


def train(model: models.Model, X_train, y_train, X_val, y_val, checkpoint_path: Path, epochs: int = 40, batch_size: int = 32):
    class_weights = get_class_weights(y_train)
    logger.info("Training %s with class weights %s", model.name, class_weights)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=default_callbacks(checkpoint_path),
        verbose=2,
    )
    return history
