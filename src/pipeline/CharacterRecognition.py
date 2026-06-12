#!/usr/bin/env python3
import os
import glob
from collections import Counter

import cv2 as cv
import numpy as np
import joblib
from sklearn import svm

IMG_SIZE = 28
SKIP_LABELS = {'BAD'}
MIN_SAMPLES_PER_CLASS = 15

# HOG descriptor over a 28x28 char window with 7x7 cells and 14x14 blocks
# at 7-pixel stride: 9 blocks x 4 cells x 9 bins = 324 features per char.
_HOG = cv.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9)


class CharacterRecognition():

    def __init__(self, C=10, gamma='scale', model_path="") -> None:
        if model_path is None or len(model_path) <= 0:
            self._clf = svm.SVC(gamma=gamma, C=C, kernel="rbf")
            self._trained = False
        else:
            self._clf = joblib.load(model_path)
            self._trained = True

    @staticmethod
    def preprocess_char(img):
        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        return _HOG.compute(img).flatten().astype(np.float32)

    @staticmethod
    def load_dataset(dirs, min_samples=MIN_SAMPLES_PER_CLASS):
        '''
        Load labeled character images organized as one directory per class.
        Filters out the reserved 'BAD' class, labels that are not a single
        alphanumeric ASCII character, and classes below min_samples so the
        classifier is trained on a clean, balanced enough set.
        '''
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
                if not (len(label) == 1 and label.isascii()
                        and label.isalnum()):
                    continue

                for img_path in glob.glob(os.path.join(label_path, '*')):
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    X.append(CharacterRecognition.preprocess_char(img))
                    y.append(label)

        X = np.array(X)
        y = np.array(y)
        if len(y) == 0:
            return X, y

        counts = Counter(y)
        kept = {lbl for lbl, c in counts.items() if c >= min_samples}
        mask = np.array([lbl in kept for lbl in y])
        return X[mask], y[mask]

    def train(self, X, y):
        self._clf.fit(X, y)
        self._trained = True
        return self

    def train_from_dirs(self, dirs):
        X, y = self.load_dataset(dirs)
        if len(y) == 0:
            raise ValueError(f"No training samples found in {dirs}")
        return self.train(X, y)

    DIGIT_SET = frozenset("0123456789")
    LETTER_SET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def predict(self, characters):
        '''
        Predict the label of each segmented character image and return the
        concatenated plate string. When the segmenter returns a count
        matching a known plate format (positions 0,1 digit, position 2
        letter, rest digit), per-position decoding restricts each label to
        the allowed class subset using the SVM's decision_function margins.
        '''
        if not self._trained:
            raise RuntimeError("CharacterRecognition has not been trained")
        if characters is None or len(characters) <= 0:
            return ""

        features = np.stack([self.preprocess_char(ch) for ch in characters])
        prior = self._format_prior_for_length(len(characters))
        if prior is None or not hasattr(self._clf, 'decision_function'):
            return "".join(self._clf.predict(features))

        scores = self._clf.decision_function(features)
        classes = self._clf.classes_
        if scores.ndim != 2 or scores.shape[1] != len(classes):
            return "".join(self._clf.predict(features))

        result = []
        for pos, allowed in enumerate(prior):
            mask = np.array([c in allowed for c in classes])
            if not mask.any():
                idx = int(np.argmax(scores[pos]))
            else:
                row = np.where(mask, scores[pos], -np.inf)
                idx = int(np.argmax(row))
            result.append(str(classes[idx]))
        return "".join(result)

    @classmethod
    def _format_prior_for_length(cls, n):
        '''
        Returns a per-position list of allowed character sets, or None for
        no prior. The two priors cover ~92% of the validation set; other
        lengths fall through to unconstrained argmax.
        '''
        if n == 8:
            return [cls.DIGIT_SET, cls.DIGIT_SET, cls.LETTER_SET,
                    cls.DIGIT_SET, cls.DIGIT_SET, cls.DIGIT_SET,
                    cls.DIGIT_SET, cls.DIGIT_SET]
        if n == 7:
            return [cls.DIGIT_SET, cls.DIGIT_SET, cls.LETTER_SET,
                    cls.DIGIT_SET, cls.DIGIT_SET, cls.DIGIT_SET,
                    cls.DIGIT_SET]
        return None

    def save(self, path):
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance._clf = joblib.load(path)
        instance._trained = True
        return instance
