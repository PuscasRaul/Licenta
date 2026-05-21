#!/usr/bin/env python3
import os
import sys
import glob
from collections import Counter
import cv2 as cv
import numpy as np
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.license_plate_extraction import DetectionPipeline as DP
from src.character_segmentation import CharacterSegmentation as CS


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dirs = [
    os.path.join(project_root, 'data', 'romanian', 'labeled'),
    os.path.join(project_root, 'data', 'characters', 'labeled'),
]

IMG_SIZE = 28
SKIP_LABELS = {'BAD'}
MIN_SAMPLES_PER_CLASS = 2


def load_dataset(dirs):
    X, y = [], []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for label_dir in os.listdir(d):
            label_path = os.path.join(d, label_dir)
            if not os.path.isdir(label_path):
                continue

            label = label_dir.upper()
            if label in SKIP_LABELS:
                continue
            # drop accidental labels like 'é', 'É'
            if not (len(label) == 1 and label.isascii() and label.isalnum()):
                continue

            for img_path in glob.glob(os.path.join(label_path, '*')):
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img.flatten())
                y.append(label)
    return np.array(X), np.array(y)


def preprocess_char(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    return (img.astype(np.float32) / 255.0).flatten()


def predict_from_path(clf, extraction_pipeline, segmentation_pipeline, path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"Not a file: {path}")
        return

    image = cv.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        return

    lp = extraction_pipeline.extraction_pipeline(image)
    if lp is None:
        print("No license plate detected")
        return

    characters = segmentation_pipeline.character_segmentation(lp)
    if not characters:
        print("No characters segmented")
        return

    features = np.stack([preprocess_char(ch) for ch in characters])
    preds = clf.predict(features)
    print(f"Predicted plate: {''.join(preds)}")

    cv.imshow("license plate", lp)
    for idx, ch in enumerate(characters):
        cv.imshow(f"{idx}:{preds[idx]}", ch)
    cv.waitKey(0)
    cv.destroyAllWindows()


X, y = load_dataset(dataset_dirs)
print(f"Loaded {len(X)} samples across {len(set(y))} classes")

counts = Counter(y)
kept = {lbl for lbl, c in counts.items() if c >= MIN_SAMPLES_PER_CLASS}
mask = np.array([lbl in kept for lbl in y])
X, y = X[mask], y[mask]
print(f"After dropping classes with <{MIN_SAMPLES_PER_CLASS} samples: "
      f"{len(X)} samples, {len(set(y))} classes")
print(f"Per-class counts: {dict(sorted(Counter(y).items()))}")

X = X.astype(np.float32) / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = svm.SVC(kernel='rbf', gamma='scale', C=10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print(f"\nAccuracy: {metrics.accuracy_score(y_test, predicted):.3f}")
print("\nClassification report:")
print(metrics.classification_report(y_test, predicted, zero_division=0))

extraction_pipeline = DP()
segmentation_pipeline = CS()

print("\nReady. Enter an image path to extract and classify characters.")
while True:
    try:
        path = input("\nImage path (empty to quit): ").strip()
    except EOFError:
        break
    if not path:
        break
    predict_from_path(clf, extraction_pipeline, segmentation_pipeline, path)
