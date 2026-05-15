"""
train.py — Train the CNN on the FER-2013 dataset.

Expected dataset layout (after Kaggle download + extraction):
  dataset/
    train/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    test/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

Run:  python train.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from model import build_emotion_model

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
TEST_DIR     = os.path.join(DATASET_DIR, "test")
MODEL_SAVE   = os.path.join("model", "emotion_model.h5")

IMG_SIZE     = 48
BATCH_SIZE   = 32   # smaller batch → less RAM usage on 8 GB machine
EPOCHS       = 25   # lightweight model converges faster; early stopping handles the rest
LR           = 1e-3

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

os.makedirs("model", exist_ok=True)

# ── Data Generators ────────────────────────────────────────────────────────────
#  Training: augment to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode="nearest",
)

#  Validation/Test: only normalise
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=EMOTIONS,
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=EMOTIONS,
    shuffle=False,
)

print(f"\nTraining samples : {train_generator.samples}")
print(f"Testing  samples : {test_generator.samples}")
print(f"Class indices    : {train_generator.class_indices}\n")

# ── Build & Compile ────────────────────────────────────────────────────────────
model = build_emotion_model(num_classes=NUM_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,          # stop earlier if no improvement — saves time
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── Training ──────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating on test set …")
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"Test  Loss     : {loss:.4f}")
print(f"Test  Accuracy : {acc:.4f}")

# ── Classification Report ─────────────────────────────────────────────────────
test_generator.reset()
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title("Confusion Matrix — FER-2013")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ── Training Curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],   label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print(f"\nModel saved to: {MODEL_SAVE}")
