import cv2 as cv
import numpy as np
from src.pipeline.HelperProcessingFunctions import HelperProcessingFunctions


class LPExtraction():

    def __init__(self, lp_aspect_ratio, lp_min_size, lp_roi=None, top_k=5):
        '''
        :param lp_aspect_ratio :tuple aspect ratio for a contour to be
        considered a license plate
        :param lp_min_size minimum area size of a region to be considered
        a candidate for lp
        :param top_k number of top-ranked LP candidates to return so the
        downstream stage can pick the best one
        '''
        self._lp_aspect_ratio = lp_aspect_ratio
        self._lp_min_size = lp_min_size
        self._lp_roi = lp_roi
        self._top_k = top_k
        self._helper = HelperProcessingFunctions()

    def mask_center(self, array: np.array) -> np.array:
        rows, cols = np.shape(array)
        mask_cols = int(cols / 6)
        mask_rows = int(rows / 2)
        top_left = (mask_cols, 0 + mask_rows)
        bottom_right = (cols - 1 - mask_cols, rows - 1)
        return self.mask(array, top_left, bottom_right)

    def mask(self, array: np.array, top_left, bottom_right) -> np.array:
        mask = np.zeros(array.shape[:2], dtype="uint8")
        cv.rectangle(mask, top_left, bottom_right, 255, -1)
        return cv.bitwise_and(array, array, mask=mask)

    def sobel(self, array: np.array) -> np.array:
        '''
        Return a 2D image sobel image
        :param array: np 2D array that represents a grayscale image
        :return Sobel image
        '''
        ddepth = cv.CV_16S
        scale = 1
        delta = 0
        grad_x = cv.Sobel(array, ddepth, 1, 0, ksize=3, scale=scale,
                          delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        return abs_grad_x

    def extraction_pipeline(self, array: np.array, top_k=None):
        '''
        Returns up to top_k LP candidate images, sorted best-first.
        '''
        lps, _ = self.extraction_pipeline_with_intermediates(array,
                                                             top_k=top_k)
        return lps

    def extraction_pipeline_with_intermediates(self, array: np.array,
                                               top_k=None):
        intermediates = {}
        if array is None:
            return None, intermediates

        k = top_k if top_k is not None else self._top_k

        if (self._lp_roi is not None):
            array = self._helper.crop_on_bounding_box(array, self._lp_roi)
        intermediates['roi'] = array

        greyscale = cv.cvtColor(array, cv.COLOR_BGR2GRAY)
        median = self._helper.median_filter(array=greyscale)
        sobel = self.sobel(median)
        thresh = self._helper.otsu_thresholding(sobel)
        thresh = cv.bitwise_not(thresh)
        intermediates['thresh'] = thresh

        open_image = self._helper.opening(thresh)
        dilated_image = self._helper.dilation(open_image)
        intermediates['dilated'] = dilated_image

        bounding_box = self._helper.find_countours(dilated_image,
                                                   self._lp_aspect_ratio,
                                                   self._lp_min_size)
        if bounding_box is None or len(bounding_box) <= 0:
            intermediates['candidates'] = bounding_box
            return None, intermediates

        bounding_box = self._merge_split_regions(bounding_box)
        intermediates['candidates'] = bounding_box

        ranked_bb = self._rank_bounding_boxes(bounding_box, thresh)
        if ranked_bb is None or len(ranked_bb) <= 0:
            return None, intermediates

        ranked_bb.sort(key=lambda item: item[1], reverse=True)
        intermediates['ranked'] = ranked_bb

        top_bbs = [item[0] for item in ranked_bb[:k]]
        intermediates['top_bbs'] = top_bbs
        intermediates['best_bb'] = top_bbs[0]

        lps = [self._deskew_plate(self._helper.crop_on_bounding_box(array, bb))
               for bb in top_bbs]
        intermediates['lps'] = lps
        intermediates['lp'] = lps[0]
        return lps, intermediates

    def _deskew_plate(self, lp_crop):
        '''
        Rotate a candidate LP crop so character baselines are horizontal.
        Uses the median angle of horizontal-ish Hough segments; if there
        is no clear skew (or it's larger than what's plausibly a tilt),
        the crop is returned unchanged.
        '''
        if lp_crop is None or lp_crop.size == 0:
            return lp_crop
        h, w = lp_crop.shape[:2]
        if h < 10 or w < 30:
            return lp_crop

        gray = cv.cvtColor(lp_crop, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 60, 180)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180,
                               threshold=max(20, w // 12),
                               minLineLength=max(15, w // 5),
                               maxLineGap=10)
        if lines is None:
            return lp_crop

        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(ang) < 25:
                angles.append(ang)

        if not angles:
            return lp_crop

        angle = float(np.median(angles))
        if abs(angle) < 0.75:
            return lp_crop

        M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv.warpAffine(lp_crop, M, (w, h),
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE)

    def _merge_split_regions(self, boxes):
        '''
        Plates often split into 2-3 fragments at the contour stage: a wide
        center gap splits a car plate horizontally, and two-line moto plates
        split vertically. Bridging that in morphology risks merging the plate
        with adjacent bumper/grille regions, so we do it here instead.

        Boxes that are spatially close and either on the same baseline (with
        similar height) or in the same column (with similar width) are
        grouped via union-find and replaced by their bounding-box union.
        A group whose union has an implausible aspect ratio is treated as
        an accidental merge: the original fragments are kept instead.
        '''
        if not boxes or len(boxes) < 2:
            return list(boxes) if boxes else []

        n = len(boxes)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(boxes[i], boxes[j]):
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(boxes[i])

        out = []
        for group in groups.values():
            if len(group) == 1:
                out.append(group[0])
                continue
            xs = [b[0] for b in group]
            ys = [b[1] for b in group]
            xes = [b[0] + b[2] for b in group]
            yes = [b[1] + b[3] for b in group]
            ux, uy = min(xs), min(ys)
            uw = max(xes) - ux
            uh = max(yes) - uy
            if uh > 0 and 1.0 <= uw / uh <= 10.0:
                out.append((ux, uy, uw, uh))
            else:
                out.extend(group)
        return out

    def _should_merge(self, a, b):
        '''
        True iff two boxes plausibly belong to the same plate. Thresholds
        are expressed as fractions of average height / width so the rule
        is scale-invariant.
        '''
        x1, y1, w1, h1 = a
        x2, y2, w2, h2 = b
        if h1 == 0 or h2 == 0 or w1 == 0 or w2 == 0:
            return False

        h_avg = (h1 + h2) / 2.0
        w_avg = (w1 + w2) / 2.0

        cy1 = y1 + h1 / 2.0
        cy2 = y2 + h2 / 2.0
        same_baseline = (abs(cy1 - cy2) < h_avg * 0.35
                         and min(h1, h2) / max(h1, h2) > 0.5)
        h_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        close_horizontally = h_gap < h_avg * 0.7
        if same_baseline and close_horizontally:
            return True

        cx1 = x1 + w1 / 2.0
        cx2 = x2 + w2 / 2.0
        same_column = (abs(cx1 - cx2) < w_avg * 0.4
                       and min(w1, w2) / max(w1, w2) > 0.4)
        v_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        close_vertically = v_gap < min(h1, h2) * 0.6
        if same_column and close_vertically:
            return True

        return False

    def _rank_bounding_boxes(self, bb, otsu):
        if bb is None:
            return None

        scored_boxes = []
        for (x, y, w, h) in bb:
            area = w * h

            roi_edges = otsu[y:y+h, x:x+w]
            edge_density = cv.countNonZero(roi_edges) / area
            scored_boxes.append(((x, y, w, h), edge_density))

        return scored_boxes
