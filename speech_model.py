"""
speech_model.py — Lightweight MLP for speech-based emotion recognition.

Input : 80 features (40 MFCC means + 40 MFCC stds) extracted from audio
Output: 7 emotion classes  (same as FER-2013 labels)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_speech_model(input_dim: int = 80, num_classes: int = 7) -> Sequential:
    """
    MLP chosen deliberately — fast to train on CPU, and MFCC features
    are already a compact 1-D representation so convolutions are not needed.
    """
    model = Sequential(name="SpeechEmotionMLP")

    model.add(Dense(256, activation="relu",
                    input_shape=(input_dim,),
                    kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    model = build_speech_model()
    model.summary()
