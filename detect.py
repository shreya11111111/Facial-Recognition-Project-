"""
detect.py — Multimodal real-time emotion recognition (Face + Speech).

Two parallel threads:
  • Main thread  : webcam → Haar cascade → CNN → face emotion probabilities
  • Audio thread : microphone → MFCC features → MLP → speech emotion probs

Fusion:
  fused_probs = 0.6 × face_probs + 0.4 × speech_probs
  (face weighted slightly higher; speech updates every 2 s)

Prerequisites:
  1.  python train.py          → model/emotion_model.h5
  2.  python train_speech.py   → model/speech_model.h5 + model/speech_norm_*.npy

If the speech model is missing, the app falls back to face-only mode.

Controls:  press  q  to quit
"""

import os
import threading
import time

import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
FACE_MODEL_PATH   = "model/emotion_model.h5"
SPEECH_MODEL_PATH = "model/speech_model.h5"
SPEECH_NORM_MEAN  = "model/speech_norm_mean.npy"
SPEECH_NORM_STD   = "model/speech_norm_std.npy"

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 48
SR          = 22050
AUDIO_DUR   = 2.0
N_MFCC      = 40
NUM_CLASSES = 7

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_COLORS = {
    "Angry":    (0,   0,   255),
    "Disgust":  (0,   128,   0),
    "Fear":     (128,   0, 128),
    "Happy":    (0,   255, 255),
    "Neutral":  (200, 200, 200),
    "Sad":      (255, 100,   0),
    "Surprise": (0,   165, 255),
}

EMOTION_EMOJI = {
    "Angry": ">:(", "Disgust": ":P", "Fear": "D:",
    "Happy": ":D",  "Neutral": ":|", "Sad":  ":(",
    "Surprise": ":O",
}

# ── Shared audio state ────────────────────────────────────────────────────────
_audio_state = {
    "probs":   np.ones(NUM_CLASSES, dtype="float32") / NUM_CLASSES,
    "emotion": "...",
    "conf":    0.0,
    "active":  False,
}
_audio_lock = threading.Lock()

# ── Keras version compatibility ──────────────────────────────────────────────
# Models trained on Google Colab (newer Keras) embed 'quantization_config'
# in every layer's config.  The locally installed Keras does not recognise
# this field and raises an error during deserialisation.  The wrappers below
# silently absorb the unknown keyword so load_model can proceed normally.
from tensorflow.keras.layers import (
    Dense as _Dense, Conv2D as _Conv2D,
    BatchNormalization as _BatchNorm, Dropout as _Dropout,
    MaxPooling2D as _MaxPool2D, Flatten as _Flatten,
    DepthwiseConv2D as _DepthConv2D,
)


def _quant_compat(base_cls):
    """Return a subclass of base_cls that silently drops quantization_config."""
    class _Compat(base_cls):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)
    _Compat.__name__ = base_cls.__name__
    _Compat.__qualname__ = base_cls.__qualname__
    return _Compat


_COMPAT_OBJECTS = {
    "Dense":              _quant_compat(_Dense),
    "Conv2D":             _quant_compat(_Conv2D),
    "BatchNormalization": _quant_compat(_BatchNorm),
    "Dropout":            _quant_compat(_Dropout),
    "MaxPooling2D":       _quant_compat(_MaxPool2D),
    "Flatten":            _quant_compat(_Flatten),
    "DepthwiseConv2D":    _quant_compat(_DepthConv2D),
}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading face model ...")
face_model = load_model(FACE_MODEL_PATH, custom_objects=_COMPAT_OBJECTS, compile=False)

speech_available = (
    os.path.exists(SPEECH_MODEL_PATH)
    and os.path.exists(SPEECH_NORM_MEAN)
    and os.path.exists(SPEECH_NORM_STD)
)

if speech_available:
    print("Loading speech model ...")
    speech_model     = load_model(SPEECH_MODEL_PATH, custom_objects=_COMPAT_OBJECTS, compile=False)
    speech_norm_mean = np.load(SPEECH_NORM_MEAN)
    speech_norm_std  = np.load(SPEECH_NORM_STD)
    print("Both models loaded — running in MULTIMODAL mode.\n")
else:
    speech_model = speech_norm_mean = speech_norm_std = None
    print("Speech model not found — running in FACE-ONLY mode.")
    print("Train it with:  python train_speech.py\n")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Voice activity detection threshold ───────────────────────────────────────
# RMS energy below this = silence / background noise → skip prediction
VAD_THRESHOLD = 0.01   # raise this if mic is very sensitive; lower if too strict


# ── Audio worker thread ───────────────────────────────────────────────────────
def audio_worker(stop_event: threading.Event) -> None:
    samples = int(AUDIO_DUR * SR)
    while not stop_event.is_set():
        try:
            audio = sd.rec(samples, samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            y = audio.flatten()

            # ── Voice Activity Detection (VAD) ─────────────────────────────
            # RMS = how loud the audio is on average
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < VAD_THRESHOLD:
                # Too quiet — person is not speaking, don't predict
                with _audio_lock:
                    _audio_state["emotion"] = "(silent)"
                    _audio_state["conf"]    = 0.0
                    _audio_state["active"]  = False
                continue

            mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            features = np.concatenate(
                [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]
            ).astype("float32")
            features = (features - speech_norm_mean) / speech_norm_std
            features = features.reshape(1, -1)

            probs = speech_model.predict(features, verbose=0)[0]
            idx   = int(np.argmax(probs))

            with _audio_lock:
                _audio_state["probs"]   = probs
                _audio_state["emotion"] = EMOTIONS[idx]
                _audio_state["conf"]    = float(probs[idx]) * 100
                _audio_state["active"]  = True

        except Exception as e:
            print(f"[Audio thread] {e}")
            time.sleep(1.0)


# ── Drawing helpers ───────────────────────────────────────────────────────────
BAR_W = 90    # narrower bars
BAR_H = 11    # shorter bar rows
TXT_SCALE = 0.34
TXT_COLOR = (230, 230, 230)


def put_text_bg(frame, text, pos, scale, color, thickness=1):
    """Draw text with a dark background rectangle for readability."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = pos
    # dark background
    cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 1),
                  (20, 20, 20), -1)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_prob_bars(frame, probs, x_start, y_start, title):
    put_text_bg(frame, title, (x_start, y_start - 3), TXT_SCALE + 0.04, (200, 200, 200))
    for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
        yy     = y_start + i * (BAR_H + 3)
        filled = int(prob * BAR_W)
        color  = EMOTION_COLORS[emo]
        # bar background
        cv2.rectangle(frame, (x_start, yy),
                      (x_start + BAR_W, yy + BAR_H), (40, 40, 40), -1)
        # filled portion
        if filled > 0:
            cv2.rectangle(frame, (x_start, yy),
                          (x_start + filled, yy + BAR_H), color, -1)
        # text label with dark bg
        label = f"{emo[:3]} {prob*100:4.1f}%"
        put_text_bg(frame, label,
                    (x_start + BAR_W + 3, yy + BAR_H - 1),
                    TXT_SCALE, TXT_COLOR)


def draw_bottom_banner(frame, face_emo, face_conf,
                       speech_emo, speech_conf, fused_emo, fused_conf,
                       multimodal):
    h, w    = frame.shape[:2]
    # solid dark strip at the bottom
    cv2.rectangle(frame, (0, h - 46), (w, h), (20, 20, 20), -1)
    cv2.line(frame, (0, h - 46), (w, h - 46), (80, 80, 80), 1)

    if multimodal:
        cv2.putText(frame, f"Face: {face_emo} ({face_conf:.0f}%)",
                    (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Speech: {speech_emo} ({speech_conf:.0f}%)",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (190, 190, 190), 1, cv2.LINE_AA)
        fused_color = EMOTION_COLORS[fused_emo]
        fused_txt   = f"FUSED: {fused_emo} {EMOTION_EMOJI[fused_emo]} ({fused_conf:.0f}%)"
        (fw, _), _ = cv2.getTextSize(fused_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, fused_txt,
                    (w // 2 - fw // 2, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fused_color, 2, cv2.LINE_AA)
    else:
        face_color = EMOTION_COLORS[face_emo]
        txt = f"Emotion: {face_emo} {EMOTION_EMOJI[face_emo]}  ({face_conf:.0f}%)"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, txt,
                    (w // 2 - tw // 2, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2, cv2.LINE_AA)


# ── Start audio thread ────────────────────────────────────────────────────────
stop_event = threading.Event()
if speech_available:
    audio_thread = threading.Thread(target=audio_worker,
                                    args=(stop_event,), daemon=True)
    audio_thread.start()
    print("Audio thread started — speak into your microphone.")

# ── Window setup (resizable + fullscreen support) ────────────────────────────
WIN_NAME = "Multimodal Emotion Recognition"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
# Start at a comfortable fixed size; user can resize or press F for fullscreen
DISPLAY_W, DISPLAY_H = 960, 540
cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

_fullscreen = False

# ── Webcam loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    stop_event.set()
    raise RuntimeError("Cannot open webcam. Check your camera index.")

print("Webcam started.  Press  q  to quit  |  f  to toggle fullscreen.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame — exiting.")
        break

    # Resize to fixed display resolution so UI elements stay proportional
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Default face state when no face is detected
    face_probs   = np.ones(NUM_CLASSES, dtype="float32") / NUM_CLASSES
    face_emotion = "Neutral"
    face_conf    = 0.0

    for (x, y, w, h) in faces:
        roi = gray[y: y + h, x: x + w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        face_probs   = face_model.predict(roi, verbose=0)[0]
        face_idx     = int(np.argmax(face_probs))
        face_emotion = EMOTIONS[face_idx]
        face_conf    = float(face_probs[face_idx]) * 100

        # ── Compute fused label for bounding box ───────────────────────────
        with _audio_lock:
            sp_probs  = _audio_state["probs"].copy()
            sp_active = _audio_state["active"]

        if speech_available and sp_active:
            fused_probs = 0.6 * face_probs + 0.4 * sp_probs
            fused_idx   = int(np.argmax(fused_probs))
            fused_label = EMOTIONS[fused_idx]
            fused_conf  = float(fused_probs[fused_idx]) * 100
            box_color   = EMOTION_COLORS[fused_label]
            box_label   = f"[FUSED] {fused_label} {EMOTION_EMOJI[fused_label]} {fused_conf:.0f}%"
        else:
            fused_label = face_emotion
            fused_conf  = face_conf
            box_color   = EMOTION_COLORS[face_emotion]
            box_label   = f"{face_emotion} {EMOTION_EMOJI[face_emotion]} {face_conf:.0f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        label_y = y - 10 if y - 10 > 10 else y + h + 22
        cv2.putText(frame, box_label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2, cv2.LINE_AA)

    # ── Left panel: face probability bars ────────────────────────────────────
    draw_prob_bars(frame, face_probs, x_start=8, y_start=22, title="FACE")

    # ── Right panel: speech probability bars ──────────────────────────────────
    if speech_available:
        with _audio_lock:
            sp_probs   = _audio_state["probs"].copy()
            sp_emotion = _audio_state["emotion"]
            sp_conf    = _audio_state["conf"]
            sp_active  = _audio_state["active"]

        frame_w = frame.shape[1]
        right_x = frame_w - BAR_W - 80
        draw_prob_bars(frame, sp_probs, x_start=right_x, y_start=22,
                       title="SPEECH" if sp_active else "SPEECH (listening...)")

        fused_probs = 0.6 * face_probs + 0.4 * sp_probs
        fused_idx   = int(np.argmax(fused_probs))
        fused_label = EMOTIONS[fused_idx]
        fused_conf  = float(fused_probs[fused_idx]) * 100

        draw_bottom_banner(frame, face_emotion, face_conf,
                           sp_emotion, sp_conf, fused_label, fused_conf,
                           multimodal=True)
    else:
        draw_bottom_banner(frame, face_emotion, face_conf,
                           "N/A", 0.0, face_emotion, face_conf,
                           multimodal=False)

    cv2.imshow(WIN_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        _fullscreen = not _fullscreen
        prop = cv2.WINDOW_FULLSCREEN if _fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)

stop_event.set()
cap.release()
cv2.destroyAllWindows()
print("Session ended.")
