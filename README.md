# Facial Expression Recognition — Major Project

Real-time emotion detection from webcam using a CNN trained on **FER-2013**.

## Emotions Detected

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## Project Structure

```
MajorProject/
├── dataset/
│   ├── train/         ← extracted from Kaggle (see Step 2)
│   └── test/
├── model/             ← created automatically during training
│   └── emotion_model.h5
├── model.py           ← CNN architecture
├── train.py           ← training + evaluation script
├── detect.py          ← live webcam inference
├── requirements.txt
└── README.md
```

---

## Step-by-Step Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FER-2013 dataset

- Go to https://www.kaggle.com/datasets/msambare/fer2013
- Click **Download**
- Extract the zip so your folder looks like:

```
dataset/
  train/
    angry/    disgust/    fear/    happy/    neutral/    sad/    surprise/
  test/
    angry/    disgust/    fear/    happy/    neutral/    sad/    surprise/
```

> **No manual cleaning needed.** The dataset is pre-sorted into folders by emotion.

### 3. Train the model

```bash
python train.py
```

- Training runs for up to 60 epochs (early stopping enabled).
- Best model auto-saved to `model/emotion_model.h5`.
- `confusion_matrix.png` and `training_curves.png` are generated after training.
- Expected accuracy: **~65–70 %** (FER-2013 is a hard dataset; human accuracy ≈ 65 %).

### 4. Run real-time detection

```bash
python detect.py
```

- Your webcam opens.
- All detected faces are outlined with a coloured bounding box.
- The predicted emotion + confidence % is displayed above each face.
- A mini probability bar for all 7 emotions is shown in the top-left corner.
- Press **q** to quit.

---

## Tips

| Situation          | Fix                                                      |
| ------------------ | -------------------------------------------------------- |
| Wrong camera opens | Change `cv2.VideoCapture(0)` → `1` or `2` in `detect.py` |
| GPU not detected   | Install `tensorflow-gpu` matching your CUDA version      |
| Low accuracy       | Increase `EPOCHS` or reduce `Dropout` in `model.py`      |

---

## Tech Stack

- **TensorFlow / Keras** — model training & inference
- **OpenCV** — webcam capture + Haar Cascade face detection
- **FER-2013** — 35,887 labelled 48×48 grayscale face images
