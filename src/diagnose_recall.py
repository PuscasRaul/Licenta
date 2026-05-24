#!/usr/bin/env python3
'''
Recall diagnostic for the LP extraction pipeline.

Runs each test image through four variants:
    strict_roi  - current production filters + bottom-middle ROI
    loose_roi   - relaxed filters         + bottom-middle ROI
    strict_full - current production filters, no ROI
    loose_full  - relaxed filters,         no ROI

For each variant we print how many bounding boxes pass the aspect/size
filters, and we save an annotated copy of the image with the top-3 boxes
drawn so the correct-plate-in-top-3 question can be checked visually.

How to read the output:
  - loose_full count >> strict_roi count
        -> the ROI and/or filter constraints are throwing the plate away.
  - all four columns are zero
        -> the candidate map itself misses the plate.
           Sobel-X is not finding it. Try black-hat morphology as a second
           signal.
  - all four are nonzero, but only loose_full's top-3 image visually
    contains the plate
        -> ranking is the issue. Edge density alone isn't enough.

Usage:
    python -m src.diagnose_recall
    python -m src.diagnose_recall --data data/valid --sample 30
'''
import os
import sys
import argparse
import random

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.license_plate_extraction import DetectionPipeline
from src.helpers import HelperProcessingFunctions

STRICT = ((2.0, 7.0), 1000)
LOOSE = ((1.5, 8.0), 500)

VARIANTS = [
    ('strict_roi',  STRICT, True),
    ('loose_roi',   LOOSE,  True),
    ('strict_full', STRICT, False),
    ('loose_full',  LOOSE,  False),
]


def bottom_mid_roi(image):
    h, w = image.shape[:2]
    return (w // 4, h // 2, w // 2, h // 2)


def run(image, aspect_size, use_roi, top_k=3):
    aspect, size = aspect_size
    roi = bottom_mid_roi(image) if use_roi else None
    pipe = DetectionPipeline(aspect, size, lp_roi=roi)
    _, inter = pipe.extraction_pipeline_with_intermediates(image, top_k=top_k)
    return inter, roi


def draw_overlay(image, inter, roi):
    out = image.copy()
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    if roi:
        cv.rectangle(out, (roi[0], roi[1]),
                     (roi[0] + roi[2], roi[1] + roi[3]),
                     (255, 255, 255), 1)
    colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255)]
    for i, (x, y, w, h) in enumerate(inter.get('top_bbs') or []):
        c = colors[i] if i < len(colors) else (255, 0, 0)
        cv.rectangle(out, (x + ox, y + oy),
                     (x + ox + w, y + oy + h), c, 2)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/test')
    parser.add_argument('--out', default='src/tests/output/diagnose')
    parser.add_argument('--sample', type=int, default=15,
                        help='Random subsample size (0 = all images)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data)
    out_dir = os.path.join(project_root, args.out)
    for sub in [v[0] for v in VARIANTS] + ['dilated']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    files = []
    for root, _, fnames in os.walk(data_dir):
        for fn in fnames:
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            files.append(os.path.join(root, fn))

    if not files:
        print(f"No images found under {data_dir}")
        return

    if 0 < args.sample < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.sample)
    files.sort()

    helper = HelperProcessingFunctions()

    print(f"Data:    {data_dir}")
    print(f"Sample:  {len(files)} images (seed={args.seed})")
    print(f"Output:  {out_dir}")
    print(f"Variants (aspect, min_size):")
    print(f"  strict = aspect {STRICT[0]}, min_size {STRICT[1]}")
    print(f"  loose  = aspect {LOOSE[0]}, min_size {LOOSE[1]}")
    print()

    header = f"{'file':32}  " + "  ".join(
        f"{v[0]:>11}" for v in VARIANTS) + f"  {'all_blobs':>11}"
    print(header)
    print('-' * len(header))

    totals = {v[0]: 0 for v in VARIANTS}
    totals['all_blobs'] = 0
    zero_strict_roi = 0
    nonzero_loose_full = 0

    for path in files:
        image = cv.imread(path)
        if image is None:
            continue
        fname = os.path.basename(path)

        per_variant = {}
        full_inter = None
        for name, ac, use_roi in VARIANTS:
            inter, roi = run(image, ac, use_roi)
            per_variant[name] = len(inter.get('candidates') or [])
            cv.imwrite(os.path.join(out_dir, name, fname),
                       draw_overlay(image, inter, roi))
            if not use_roi:
                full_inter = inter

        all_blobs = []
        if full_inter is not None and 'dilated' in full_inter:
            all_blobs = helper.find_countours(
                full_inter['dilated'], None, None) or []
            cv.imwrite(os.path.join(out_dir, 'dilated', fname),
                       full_inter['dilated'])

        cols = dict(per_variant)
        cols['all_blobs'] = len(all_blobs)

        line = f"{fname[:32]:32}  " + "  ".join(
            f"{cols[v[0]]:>11}" for v in VARIANTS)
        line += f"  {cols['all_blobs']:>11}"
        print(line)

        for k, n in cols.items():
            totals[k] += n
        if cols['strict_roi'] == 0:
            zero_strict_roi += 1
        if cols['loose_full'] > 0:
            nonzero_loose_full += 1

    print('-' * len(header))
    line = f"{'TOTAL candidates':32}  " + "  ".join(
        f"{totals[v[0]]:>11}" for v in VARIANTS)
    line += f"  {totals['all_blobs']:>11}"
    print(line)
    print()
    print(f"Images with zero strict_roi candidates: "
          f"{zero_strict_roi}/{len(files)}")
    print(f"Images with any loose_full candidate:   "
          f"{nonzero_loose_full}/{len(files)}")
    print()
    print(f"Annotated outputs saved per-variant under {out_dir}/")
    print("Open strict_roi/ and loose_full/ side-by-side to see where the "
          "right plate lives.")


if __name__ == '__main__':
    main()
