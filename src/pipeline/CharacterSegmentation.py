import cv2 as cv
import numpy as np
from src.pipeline.HelperProcessingFunctions import HelperProcessingFunctions


class CharacterSegmentation():
    def __init__(self) -> None:
        self._helper = HelperProcessingFunctions()

    def character_segmentation(self, license_plate):
        '''
        Accepts a single LP array or a list of LP candidates (best-first).
        When given multiple candidates, runs segmentation on each and
        returns the character set from the highest-scoring candidate.
        '''
        if license_plate is None:
            return None

        candidates = (license_plate if isinstance(license_plate, list)
                      else [license_plate])

        best_chars = None
        best_score = -float('inf')
        for lp in candidates:
            if lp is None or lp.size == 0:
                continue
            chars, bboxes = self._segment(lp)
            if chars is None or len(chars) == 0:
                continue
            score = self._score(bboxes, lp.shape[0])
            if score > best_score:
                best_score = score
                best_chars = chars

        return best_chars

    def _segment(self, license_plate: np.array):
        body = self._tighten_plate_crop(license_plate)
        if body is None or body.size == 0:
            return None, None
        H, W = body.shape[:2]
        if H < 8 or W < 20:
            return None, None

        greyscale = (body if body.ndim == 2
                     else cv.cvtColor(body, cv.COLOR_BGR2GRAY))

        block = 11
        thresholded = cv.adaptiveThreshold(
            greyscale,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            block,
            2)

        contours = self._helper.find_countours(thresholded,
                                               aspect_ratio_bounds=(0.04, 2.0))
        if contours is None:
            return None, None

        min_height = 0.40 * H
        max_height = 0.95 * H
        min_w_h = 0.08
        max_w_h = 1.3
        border_margin = max(1, int(0.015 * W))

        stds = []
        for cnt in contours:
            crop = self._helper.crop_on_bounding_box(greyscale, cnt)
            stds.append(cv.meanStdDev(crop)[1].item() if crop.size > 0 else 0.0)

        if not stds:
            return None, None

        stds_arr = np.array(stds, dtype="uint8")
        otsu_thr, _ = cv.threshold(stds_arr, 0, 255,
                                   cv.THRESH_BINARY + cv.THRESH_OTSU)
        std_thr = max(15.0, float(otsu_thr))

        kept = []
        for c, s in zip(contours, stds):
            x, y, w, h = c
            if s < std_thr:
                continue
            if not (min_height <= h <= max_height):
                continue
            wh = w / h
            if not (min_w_h <= wh <= max_w_h):
                continue
            if x < border_margin or (x + w) > (W - border_margin):
                continue
            kept.append((x, y, w, h))

        if not kept:
            return None, None

        kept.sort(key=lambda b: b[0])

        heights = np.array([h for _, _, _, h in kept], dtype=np.float32)
        med_h = float(np.median(heights))
        kept = [b for b in kept
                if 0.55 * med_h <= b[3] <= 1.45 * med_h]
        if not kept:
            return None, None

        kept = self._split_wide_blobs(kept, thresholded)

        characters = [self._helper.crop_on_bounding_box(body, bb)
                      for bb in kept]
        return characters, kept

    def _tighten_plate_crop(self, lp):
        '''
        Crop to the largest light-colored connected component (the plate
        body). Falls back to the original crop when the result is not
        plausibly a plate, so we never produce something worse than the
        input.
        '''
        if lp is None or lp.size == 0:
            return lp
        gray = (lp if lp.ndim == 2 else cv.cvtColor(lp, cv.COLOR_BGR2GRAY))
        _, bw = cv.threshold(gray, 0, 255,
                             cv.THRESH_BINARY + cv.THRESH_OTSU)
        n, _, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 1:
            return lp
        areas = stats[1:, cv.CC_STAT_AREA]
        idx = 1 + int(np.argmax(areas))
        x, y, w, h, _ = stats[idx]
        H, W = gray.shape[:2]
        if w * h < 0.35 * H * W:
            return lp
        if w / max(1, h) < 1.5:
            return lp
        pad = max(2, min(h, w) // 20)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(W - x, w + 2 * pad)
        h = min(H - y, h + 2 * pad)
        return lp[y:y + h, x:x + w]

    def _split_wide_blobs(self, boxes, binarized):
        '''
        For blobs noticeably wider than tall (likely touching characters),
        find column-projection valleys and split at them.
        '''
        result = []
        for x, y, w, h in boxes:
            if w < 1.2 * h:
                result.append((x, y, w, h))
                continue
            roi = binarized[y:y + h, x:x + w]
            if roi.size == 0:
                result.append((x, y, w, h))
                continue
            expected_w = max(1, int(0.55 * h))
            n_splits = max(2, min(10, int(round(w / expected_w))))
            col_sum = roi.sum(axis=0)
            seg_w = w // n_splits
            split_xs = []
            for i in range(1, n_splits):
                lo = max(0, i * seg_w - seg_w // 3)
                hi = min(w, i * seg_w + seg_w // 3)
                if hi <= lo:
                    continue
                valley = lo + int(np.argmin(col_sum[lo:hi]))
                split_xs.append(valley)
            if not split_xs:
                result.append((x, y, w, h))
                continue
            prev = 0
            for s in split_xs + [w]:
                width = s - prev
                if width > 3:
                    result.append((x + prev, y, width, h))
                prev = s
        return result

    def _score(self, bboxes, lp_height):
        '''
        Score a segmentation result; higher is better. A plate that yields
        a count close to the typical 6-8 chars with consistent heights and
        baseline alignment is preferred.
        '''
        if not bboxes:
            return -float('inf')

        n = len(bboxes)
        if n < 3:
            return -float('inf')

        heights = np.array([h for _, _, _, h in bboxes], dtype=np.float32)
        ys = np.array([y for _, y, _, _ in bboxes], dtype=np.float32)
        mean_h = float(heights.mean()) + 1e-6

        h_consistency = -float(heights.std()) / mean_h
        y_consistency = -float(ys.std()) / mean_h

        ideal = 7
        count_score = -abs(n - ideal) * 0.5

        return count_score + h_consistency + y_consistency
