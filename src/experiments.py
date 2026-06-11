#!/usr/bin/env python3
'''
Experiment harness for chapter 5.7.

Runs four studies on the annotated localization set (data/test) and the
annotated character set (data/LP-characters):

  1. Ablation on the top_k localization candidates (k = 1 vs the default).
  2. Ablation on the deskew correction (enabled vs disabled).
  3. Ablation on the merge-split-regions step (enabled vs disabled).
  4. Comparison of projection-based and contour-based segmentation.

Each variant reports the same triplet of metrics emitted by evaluate.py
(localization, segmentation, end-to-end), so deltas attributable to one
component are directly comparable.
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.LPExtraction import LPExtraction
from src.pipeline.CharacterSegmentation import CharacterSegmentation
from src.evaluate import (
    evaluate_localization,
    evaluate_recognition,
    load_recognition,
    _ratio,
)


def _print_row(name, loc, seg, e2e):
    def fmt(stats):
        return f"{stats.correct:>4}/{stats.total:<4} ({_ratio(stats):.3f})"
    print(f"  {name:<32} loc {fmt(loc)}   seg {fmt(seg)}   "
          f"e2e {fmt(e2e)}")


def _run(extraction, segmentation, recognition):
    loc = evaluate_localization(extraction)
    seg, e2e = evaluate_recognition(segmentation, recognition)
    return loc, seg, e2e


class _PatchedExtraction(LPExtraction):
    '''
    Subclass that selectively bypasses parts of the pipeline so we can
    isolate their contribution. Patched stages return their input
    unchanged.
    '''

    def __init__(self, *args, skip_deskew=False, skip_merge=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_deskew = skip_deskew
        self._skip_merge = skip_merge

    def _deskew_plate(self, lp_crop):
        if self._skip_deskew:
            return lp_crop
        return super()._deskew_plate(lp_crop)

    def _merge_split_regions(self, boxes):
        if self._skip_merge:
            return list(boxes) if boxes else []
        return super()._merge_split_regions(boxes)


class _ContourSegmentation(CharacterSegmentation):
    '''
    Forces the legacy contour-based segmentation path on every candidate,
    so we can compare it against the projection-based default.
    '''

    def _segment(self, license_plate):
        return self._segment_contours(license_plate)


def main():
    recognition = load_recognition()
    seg_default = CharacterSegmentation()

    print("== Baseline ==")
    extraction = LPExtraction((1.5, 8.0), 500)
    loc, seg, e2e = _run(extraction, seg_default, recognition)
    _print_row("baseline (top_k=5)", loc, seg, e2e)

    print("\n== Ablation: top_k ==")
    extraction = LPExtraction((1.5, 8.0), 500, top_k=1)
    loc, seg, e2e = _run(extraction, seg_default, recognition)
    _print_row("top_k=1", loc, seg, e2e)

    print("\n== Ablation: deskew ==")
    extraction = _PatchedExtraction((1.5, 8.0), 500, skip_deskew=True)
    loc, seg, e2e = _run(extraction, seg_default, recognition)
    _print_row("deskew disabled", loc, seg, e2e)

    print("\n== Ablation: _merge_split_regions ==")
    extraction = _PatchedExtraction((1.5, 8.0), 500, skip_merge=True)
    loc, seg, e2e = _run(extraction, seg_default, recognition)
    _print_row("merge disabled", loc, seg, e2e)

    print("\n== Comparison: segmentation method ==")
    extraction = LPExtraction((1.5, 8.0), 500)
    seg_contours = _ContourSegmentation()
    loc, seg, e2e = _run(extraction, seg_contours, recognition)
    _print_row("contour-based segmentation", loc, seg, e2e)


if __name__ == '__main__':
    main()
