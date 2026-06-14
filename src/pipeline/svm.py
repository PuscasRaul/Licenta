#!/usr/bin/env python3
import os
import sys
from collections import Counter

import joblib
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.CharacterRecognition import CharacterRecognition

project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
dataset_dirs = [os.path.join(project_root, 'data', 'labeled')]
model_out = os.path.join(project_root, 'data', 'svm_model.joblib')

X, y = CharacterRecognition.load_dataset(dataset_dirs)
print(f"Loaded {len(X)} samples across {len(set(y))} classes")
print(f"Per-class counts: {dict(sorted(Counter(y).items()))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = svm.SVC(kernel='rbf', gamma='scale', C=10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print(f"\nAccuracy: {metrics.accuracy_score(y_test, predicted):.3f}")
print("\nClassification report:")
print(metrics.classification_report(y_test, predicted, zero_division=0))

labels = sorted(set(y_test) | set(predicted))
cm = metrics.confusion_matrix(y_test, predicted, labels=labels)
print("\nConfusion matrix (rows = true, cols = predicted):")
header = "    " + " ".join(f"{lbl:>3}" for lbl in labels)
print(header)
for lbl, row in zip(labels, cm):
    print(f"{lbl:>3} " + " ".join(f"{v:>3}" for v in row))

joblib.dump(clf, model_out)
print(f"\nModel saved to {model_out}")
