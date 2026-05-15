"""
model.py — CNN architecture for FER-2013 facial expression recognition.
Input: 48x48 grayscale images  |  Output: 7 emotion classes
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.regularizers import l2


def build_emotion_model(num_classes: int = 7) -> Sequential:
    """
    Lightweight CNN for CPU-only machines (i3 / 8 GB RAM).
    3 blocks with small filter counts keep training under ~1.5 hrs.

    Block 1-3 : Conv(32/64/128) → BN → MaxPool → Dropout
    Head       : Flatten → Dense(128) → BN → Dropout → Dense(num_classes)
    """
    model = Sequential(name="EmotionCNN_Lite")

    # ── Block 1 ── 32 filters ─────────────────────────────
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu",
                     input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ── 64 filters ─────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ── 128 filters ────────────────────────────
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Classifier Head ──────────────────────────────────
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    model = build_emotion_model()
    model.summary()
