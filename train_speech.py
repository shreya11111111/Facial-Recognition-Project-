"""
train_speech.py — Train speech emotion model on RAVDESS dataset.

Dataset  : RAVDESS Emotional Speech Audio
Kaggle   : https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Expected layout after extraction:
  ravdess/
    Actor_01/
      03-01-01-01-01-01-01.wav
      ...
    Actor_02/
    ...  (24 actors total)

RAVDESS filename format:  MM-VV-EE-II-SS-RR-AA.wav
  EE (emotion code):
    01=neutral  02=calm  03=happy  04=sad
    05=angry    06=fearful  07=disgust  08=surprised

Mapped to FER-2013 labels:
  angry(0) disgust(1) fear(2) happy(3) neutral(4) sad(5) surprise(6)
  calm → neutral  (no FER equivalent)

Run:  python train_speech.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from speech_model import build_speech_model

# ── Config ────────────────────────────────────────────────────────────────────
RAVDESS_DIR  = "ravdess"
MODEL_SAVE   = os.path.join("model", "speech_model.h5")
SR           = 22050          # sample rate
DURATION     = 2.5            # seconds to load per clip
N_MFCC       = 40             # MFCC coefficients
INPUT_DIM    = N_MFCC * 2    # mean + std = 80 features
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3

os.makedirs("model", exist_ok=True)

# RAVDESS emotion code → FER-2013 index mapping
RAVDESS_TO_FER = {
    "01": 4,   # neutral  → neutral
    "02": 4,   # calm     → neutral  (closest match)
    "03": 3,   # happy    → happy
    "04": 5,   # sad      → sad
    "05": 0,   # angry    → angry
    "06": 2,   # fearful  → fear
    "07": 1,   # disgust  → disgust
    "08": 6,   # surprised→ surprise
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(file_path: str) -> np.ndarray:
    """Load audio clip and return 80-dim MFCC feature vector."""
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)

    # Pad if clip is shorter than DURATION
    target_len = int(SR * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


# ── Load dataset ──────────────────────────────────────────────────────────────
print("Scanning RAVDESS audio files …")
wav_files = glob.glob(os.path.join(RAVDESS_DIR, "**", "*.wav"), recursive=True)

if not wav_files:
    raise FileNotFoundError(
        f"No .wav files found under '{RAVDESS_DIR}/'. "
        "Download RAVDESS from Kaggle and extract it as described in the docstring."
    )

print(f"Found {len(wav_files)} audio files. Extracting features …")

X, y = [], []
skipped = 0

for path in wav_files:
    basename = os.path.basename(path)
    parts = basename.replace(".wav", "").split("-")
    if len(parts) < 3:
        skipped += 1
        continue

    emo_code = parts[2]
    if emo_code not in RAVDESS_TO_FER:
        skipped += 1
        continue

    try:
        features = extract_features(path)
        X.append(features)
        y.append(RAVDESS_TO_FER[emo_code])
    except Exception as e:
        print(f"  Skipped {basename}: {e}")
        skipped += 1

print(f"Loaded {len(X)} samples  ({skipped} skipped)\n")

X = np.array(X, dtype="float32")           # (N, 80)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# ── Normalise features ────────────────────────────────────────────────────────
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_norm = (X - mean) / std

# Save normalisation stats — needed at inference time
np.save("model/speech_norm_mean.npy", mean)
np.save("model/speech_norm_std.npy",  std)
print("Normalisation stats saved to model/speech_norm_mean.npy & speech_norm_std.npy")

# ── Train / validation split ──────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, y_cat, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}\n")

# ── Class weights (fix neutral over-representation from calm+neutral mapping) ──
y_ints = np.array(y)
cls_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_ints),
    y=y_ints,
)
class_weight_dict = dict(enumerate(cls_weights))
print("Class weights:", {EMOTIONS[k]: round(v, 2) for k, v in class_weight_dict.items()})

# ── Build & compile ───────────────────────────────────────────────────────────
model = build_speech_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,   # penalises neutral less, rare classes more
    callbacks=callbacks,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=1)
print(f"\nVal Loss     : {loss:.4f}")
print(f"Val Accuracy : {acc:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title("Confusion Matrix — Speech Model (RAVDESS)")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("speech_confusion_matrix.png")
plt.show()

print(f"\nSpeech model saved to: {MODEL_SAVE}")
