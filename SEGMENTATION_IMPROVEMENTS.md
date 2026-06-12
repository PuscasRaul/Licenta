# Character Segmentation Improvements

Branch: `improve-character-segmentation`
Baseline commit: `9e7990d` (main)
Improvement commit: `c41672c`

## Context

Baseline metrics were obtained by running `python -m src.evaluate` over `data/VALIDATE` (203 images, 186 with ground-truth plate text) and then `python -m src.metrics` to compare against the existing `data/evaluate_output/comparison_new` ground-truth column.

The confusion matrix (Levenshtein-aligned, `gt -> pred : count`) showed two distinct failure modes:

- **Character drops** (~250 total) dominated by `5 -> ∅: 55`, `1 -> ∅: 48`, `3 -> ∅: 20`, `9 -> ∅: 17`, `4 -> ∅: 16`, `8 -> ∅: 15`, `2 -> ∅: 15`, `F -> ∅: 13`.
- **Spurious insertions** (~30 total) dominated by `∅ -> A: 10`, `∅ -> T: 4`, `∅ -> H: 4`, `∅ -> B: 3`, `∅ -> L: 3` — consistent with frame artefacts (EU strip, plate frame, mounting bolts) being mistaken for characters.

OCR substitutions (e.g. `D -> 0: 4`, `4 -> F: 6`) were comparatively rare and are out of scope for segmentation work.

## Results

| Metric | Before | After | Δ |
|---|---|---|---|
| **E2E exact match** | **48.9 %** | **59.7 %** | **+10.8 pp** |
| E2E correct plates | 91 / 186 | 111 / 186 | +20 |
| OCR Levenshtein similarity | 0.768 | 0.801 | +0.033 |
| Segmentation count agreement | 0.809 | 0.839 | +0.030 |
| Localization IoU (untouched) | 0.588 | 0.588 | 0 |
| Confusion matrix size | 84 pairs | 83 pairs | -1 |

Insertions in the confusion matrix dropped from ~30 to ~5.

## Changes

All changes are in `src/pipeline/CharacterSegmentation.py`.

### 1. Plate-body tightening (`_tighten_plate_crop`)

Before binarization, the plate candidate is tightened to the largest light-coloured connected component (Otsu on the candidate, then `cv.connectedComponentsWithStats`). A small padding (`min(h, w) // 20`) is added so character edges are not clipped.

The result is rejected if the candidate component covers less than 35 % of the crop area or has aspect ratio below 1.5:1 — in those cases the original LP crop is returned unchanged. This guards against degenerate cases where the heuristic would shrink to the wrong region.

**Why:** loose LP crops (mean IoU 0.588) routinely included the EU strip on the left, the painted plate frame, and the mounting bolts at the corners. These produced character-shaped contours that survived all downstream filters. Tightening the crop to the actual plate body eliminates most of them before they can be detected.

### 2. Lowered `min_w_h` from 0.15 to 0.08

The width-to-height ratio floor for kept contours was lowered. Real `1` characters have w/h in the range 0.07–0.12 and were being rejected wholesale by the previous 0.15 floor.

**Why:** `1 -> ∅: 48` was the second-largest drop category, with the geometric filter as the direct cause.

### 3. Border-touching rejection

Contours whose bounding box starts within `max(1, int(0.015 * W))` of the left edge or ends within the same margin of the right edge are rejected.

**Why:** even after plate-body tightening, occasional frame remnants survive at the left/right edges. They produce the `∅ -> A`, `∅ -> T`, etc. insertion patterns.

### 4. Median-height consistency pass

After the per-contour filters, the median height of surviving boxes is computed and boxes outside `[0.55 * med_h, 1.45 * med_h]` are dropped.

**Why:** when a few large noise blobs slip through the absolute height filter (`0.40 * H <= h <= 0.95 * H`), they can be identified relative to the median character height of the real survivors.

### 5. Vertical-projection wide-blob splitter (`_split_wide_blobs`)

For boxes wider than `1.2 * h`, the column-sum projection of the binarized region inside the box is computed. The box is split into `round(w / (0.55 * h))` equal-width segments and the actual split position in each segment is chosen at the column with the lowest projection value (the deepest local valley).

**Why:** touching characters (e.g. `11`, `00`) are detected as a single wide contour. Splitting at projection valleys recovers the constituent characters without depending on the binarization to break them apart.

### 6. Contour aspect-ratio bounds widened

The `find_countours` call now accepts `aspect_ratio_bounds=(0.04, 2.0)` (was `(0, 1)`). The wider upper bound lets touching-character blobs through to the splitter; the explicit lower bound keeps obviously-degenerate boxes out.

## Discarded experiments

These were tried during development and reverted because they did not help or regressed metrics. They are documented here to save anyone repeating them.

- **3×3 morphological close** after binarization (intended to reconnect broken thin strokes). On 27-pixel-tall plates, the 3-pixel kernel bridged the gaps between adjacent characters and reduced the raw contour count from 23 to 4 in the worst observed case.
- **(2,1) vertical-only close** with the same goal. Smaller effect than 3×3 but still reduced the contour count from 26 to 17 on the same test plate. Net negative.
- **Scaled `block_size = max(11, (H // 2) | 1)`** for the adaptive threshold. Made the threshold more permissive on tall plates but added no measurable improvement at the dataset's typical plate sizes (12–30 px tall).
- **`C = 5`** in `cv.adaptiveThreshold` (was 2). Made the binarization more conservative and dropped character coverage. Reverted to `C = 2`.
- **Looser count tolerance in `_score`** (`count_score = 0 if 6 <= n <= 8`). Equal E2E (59.7 %), slightly lower OCR similarity (0.796 vs 0.801), and a larger confusion matrix. Reverted to the original `-abs(n - 7) * 0.5`.

## Known remaining issues

The `5 -> ∅` drop count is essentially unchanged (55 vs 55) despite the other improvements. Tracing several failure cases (`51F88270 -> 1F88270`, `52Y6490 -> 2T6490`, `56N0666 -> 6N0666`, `51A227 -> 227`) showed the leading `5` is **missing from the LP crop itself** — it never reaches `_segment`. This is a localization clipping issue, not a segmentation one, and would need changes in `src/pipeline/LPExtraction.py` to address.

Substitutions like `D -> 0`, `4 -> F`, `Y -> T/J` are OCR-stage errors and live in `src/pipeline/CharacterRecognition.py`.
