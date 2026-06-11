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
        greyscale = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        thresholded = cv.adaptiveThreshold(
            greyscale,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            11,
            2)
        contours = self._helper.find_countours(thresholded,
                                               aspect_ratio_bounds=(0, 1))

        if contours is None:
            return None, None

        min_height = 0.40 * license_plate.shape[0]
        max_height = 0.90 * license_plate.shape[0]

        stds = []
        for cnt in contours:
            crop = self._helper.crop_on_bounding_box(greyscale, cnt)
            stds.append(cv.meanStdDev(crop)[1].item())

        if not stds:
            return None, None

        stds_arr = np.array(stds, dtype="uint8")
        thr, _ = cv.threshold(stds_arr, 0, 255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU)

        selected_cnt = sorted(
            (c for c, s in zip(contours, stds)
             if s > thr and min_height <= c[3] <= max_height),
            key=lambda bb: bb[0])

        characters = [self._helper.crop_on_bounding_box(license_plate, cnt)
                      for cnt in selected_cnt]
        return (characters, selected_cnt)

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
