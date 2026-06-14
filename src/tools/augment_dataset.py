#!/usr/bin/env python3
'''
Augment the labeled character dataset with small in-plane rotations.

Runtime characters reach the classifier with slight tilts and small
boundary perturbations (segmentation does not perfectly axis-align the
crops), while the labeled set is dominated by near-axis-aligned samples.
Adding rotated copies of each original broadens the in-class variance
the SVM learns, reducing same-shape substitutions (Y-M, F-6, G-4)
without changing the classifier itself.

Outputs are written next to the originals with a suffix marker so the
script is idempotent across re-runs and so the augmented files can be
distinguished from real samples for later cleanup if needed.

Usage:
    conda run -n licenta python -m src.tools.augment_dataset
'''
import os
import sys

import cv2 as cv
import numpy as np

ANGLES = (-5, -3, 3, 5)
TAG = '_aug_rot'
SKIP_CLASSES = {'BAD'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def rotate(img, angle):
    '''
    Rotate around the image center. The border is filled with the
    median of the four corner pixels, an approximation of the local
    background that matches how preprocess_char fills its padding.
    '''
    h, w = img.shape[:2]
    corners = (int(img[0, 0]), int(img[0, -1]),
               int(img[-1, 0]), int(img[-1, -1]))
    fill = int(np.median(corners))
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv.warpAffine(
        img, M, (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=fill)


def angle_tag(angle):
    sign = 'p' if angle >= 0 else 'n'
    return f'{TAG}_{sign}{abs(angle):02d}'


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    labeled_dir = os.path.join(project_root, 'data', 'labeled')
    if not os.path.isdir(labeled_dir):
        print(f'Not found: {labeled_dir}', file=sys.stderr)
        sys.exit(1)

    total_originals = 0
    total_written = 0
    total_skipped_existing = 0

    for class_name in sorted(os.listdir(labeled_dir)):
        class_dir = os.path.join(labeled_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name.upper() in SKIP_CLASSES:
            continue

        class_originals = 0
        class_written = 0
        for fname in sorted(os.listdir(class_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTS:
                continue
            if TAG in stem:
                continue

            src_path = os.path.join(class_dir, fname)
            img = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            class_originals += 1

            for angle in ANGLES:
                out_name = f'{stem}{angle_tag(angle)}{ext}'
                out_path = os.path.join(class_dir, out_name)
                if os.path.exists(out_path):
                    total_skipped_existing += 1
                    continue
                rotated = rotate(img, angle)
                cv.imwrite(out_path, rotated)
                class_written += 1

        total_originals += class_originals
        total_written += class_written
        print(f'  {class_name}: {class_originals} originals '
              f'-> {class_written} new variants')

    print(f'\nProcessed {total_originals} originals across all classes; '
          f'wrote {total_written} variants '
          f'(skipped {total_skipped_existing} already present).')


if __name__ == '__main__':
    main()
