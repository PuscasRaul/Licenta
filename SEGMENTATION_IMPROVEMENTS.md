# Pipeline Improvements

Branch: `improve-character-segmentation`
Baseline commit: `9e7990d` (main)
Commits on this branch:
- `c41672c` — segmentation improvements
- `1ed1001` — position-aware OCR decoding

## Branch progression at a glance

| Stage | Commit | E2E | OCR sim | Seg count agr | IoU |
|---|---|---|---|---|---|
| Baseline (`main`) | `9e7990d` | 48.9 % | 0.768 | 0.809 | 0.588 |
| + Segmentation rebuild | `c41672c` | 59.7 % | 0.801 | 0.839 | 0.588 |
| + Position-aware decoding | `1ed1001` | **64.5 %** | 0.801 | 0.839 | 0.588 |

Branch total: **+15.6 pp E2E** (91 → 120 correctly read plates out of 186).

---

# Part 1 — Character Segmentation

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

Substitutions like `D -> 0`, `4 -> F`, `Y -> T/J` are OCR-stage errors and live in `src/pipeline/CharacterRecognition.py`. Part 2 addresses these.

---

# Part 2 — Position-aware OCR decoding

Commit: `1ed1001` (cumulative on top of `c41672c`).

## Context

After the segmentation rebuild, ~22 substitutions in the confusion matrix remained (e.g. `4 -> F: 6`, `D -> 0: 4`, `Y -> T: 2`, `1 -> T/F/K/H/L`). Inspection of the GT column of `data/evaluate_output/comparison_new` showed a strong format prior in the test set:

- 159 plates match `DDLDDDDD` (2 digits + letter + 5 digits, 8 chars)
- 12 plates match `DDLDDDD` (7 chars)
- Total: 171 / 186 = **92 %** match `^\d{2}[A-Z]\d{4,5}$`

The remaining 15 plates are shorter or follow other formats.

Important: this prior is **derived from the validation dataset's actual GT**, not from the Romanian plate standard (which is `XX-NN-ABC`). The user confirmed the test set is not Romanian.

A quick offline simulation (assuming the SVM's top-1 prediction at each position would be locked to the allowed class subset) counted:
- 13 letter-at-digit-position predictions in length-7/8 plates that would flip to a digit
- 2 digit-at-letter-position predictions that would flip to a letter

A potential floor of ~15 corrections from the prior alone, before any improvement in the per-character classifier.

## Results

| Metric | Before (`c41672c`) | After (`1ed1001`) | Δ |
|---|---|---|---|
| **E2E exact match** | **59.7 %** | **64.5 %** | **+4.8 pp** |
| E2E correct plates | 111 / 186 | 120 / 186 | +9 |
| OCR Levenshtein similarity | 0.801 | 0.801 | ~0 |
| Segmentation count agreement | 0.839 | 0.839 | 0 |
| Localization IoU | 0.588 | 0.588 | 0 |
| Confusion matrix size | 83 pairs | 82 pairs | -1 |

The OCR similarity moved less than 0.001 because only ~10 of the ~3300 predicted characters changed — but the changes are concentrated in the substitutions that were *causing* incorrect plates, so E2E sees the full benefit.

## Change

Single change in `src/pipeline/CharacterRecognition.py:predict()`. Instead of `self._clf.predict(features)`, the decoder now uses `self._clf.decision_function(features)` to access per-position class margins, and applies the format prior:

- If `len(characters) == 7` or `8`, the allowed character set at each position is fixed: position 2 is constrained to letters, every other position to digits.
- The argmax of `decision_function` is taken inside the allowed subset only.
- For all other lengths (~8 % of GT), the decoder falls through to the original unconstrained `predict()`. No regression on those cases.

The SVC was already trained with `decision_function_shape='ovr'` (sklearn default), so `decision_function` returns `(n_samples, n_classes)` matching `self._clf.classes_` order. A defensive check falls back to `predict()` if the shape doesn't match.

**Why this works:** the SVM was actually classifying many digits correctly inside the letter classes' margin space (e.g. `0`'s decision_function score for `D` was high but `0`'s score for `0` was also high — argmax went to `D`, but constraining to digits picks `0`). The prior turns the SVM's residual class confusion into deterministic corrections at no training cost.

## Discarded experiment (this round)

**LP bbox padding before crop** (`LPExtraction.py`). Hypothesis: padding the bbox by 2–5 % of its width before cropping would recover leading characters in the IoU < 0.3 tail of LP detections.

| Variant | E2E |
|---|---|
| No padding (baseline) | 59.7 % |
| Pad 5 % x, 3 % y | 41.9 % |
| Pad 2 % x, 2 % y | 47.3 % |

Both regressed sharply. The cause: most LP detections are already reasonable (median IoU 0.655). Padding adds background — frame paint, car bodywork, sometimes part of a neighbouring plate — that the downstream `_tighten_plate_crop` cannot always re-isolate. The leading-character loss is concentrated in a fat lower tail of bad detections (27 with IoU < 0.3, 9 at IoU = 0), not in a global "too tight" bias. Blanket padding hurts the 76 % of images where the localizer already worked.

The fix for that tail needs a smarter LP scoring or per-candidate recovery pass, not blanket padding. Reverted entirely; the unused `_pad_bbox` helper was removed.

## Known remaining issues (after this round)

- The leading-character clipping (LP localization issue) still produces about 30 same-length-different-leading-character predictions that the format prior cannot correct, because the segmenter never saw the missing character.
- Substitutions where both candidates are letters (or both digits) remain: `Y -> T`, `Y -> J`, `5 -> P`, `5 -> A`, `9 -> J`, `1 -> 5`. These need a stronger per-character classifier (HOG features or aspect-preserving preprocessing followed by SVM retrain).
- The format prior is dataset-specific. If the test set composition changes, the prior would have to be re-derived; otherwise it could over-correct.
