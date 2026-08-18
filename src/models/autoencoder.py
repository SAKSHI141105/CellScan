"""Autoencoders for anomaly-detection-flavored unsupervised learning.

Same idea in both pipelines: train the autoencoder to reconstruct *benign*
samples well, then use reconstruction error on unseen samples as a
malignancy signal — high error means "this doesn't look like what I learned
benign tissue looks like." We deliberately train on benign-only data; training
on the full mixed set would just teach it to reconstruct everything equally
well and the anomaly signal disappears.
"""
from __future__ import annotations

import numpy as np
from tensorflow.keras import layers, models, optimizers

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_dense_autoencoder(input_dim: int, encoding_dim: int = 8) -> models.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(24, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    bottleneck = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(24, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs, name="tabular_autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    return autoencoder


def build_conv_autoencoder(img_size: int = 224, channels: int = 1) -> models.Model:
    inputs = layers.Input(shape=(img_size, img_size, channels))
    x = layers.Conv2D(32, 3, activation="relu", padding="same", strides=2)(inputs)
    x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)
    encoded = layers.Conv2D(8, 3, activation="relu", padding="same", strides=2, name="bottleneck")(x)

    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(encoded)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same", strides=2)(x)

    autoencoder = models.Model(inputs, x, name="image_autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    return autoencoder


def fit_on_benign_only(autoencoder: models.Model, X: np.ndarray, y_benign_mask: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.15):
    from tensorflow.keras.callbacks import EarlyStopping

    X_benign = X[y_benign_mask]
    logger.info("Training autoencoder on %d benign-only samples", len(X_benign))
    history = autoencoder.fit(
        X_benign, X_benign,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0,
    )
    return history


def reconstruction_error(autoencoder: models.Model, X: np.ndarray) -> np.ndarray:
    reconstructed = autoencoder.predict(X, verbose=0)
    axes = tuple(range(1, X.ndim))
    return np.mean(np.square(X - reconstructed), axis=axes)


def anomaly_threshold(errors_on_benign: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(errors_on_benign, percentile))
