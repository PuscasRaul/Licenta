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

        greyscale = (array if array.ndim == 2
                     else cv.cvtColor(array, cv.COLOR_BGR2GRAY))
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

        bounding_box = self._merge_candidates(bounding_box)
        intermediates['candidates'] = bounding_box

        ranked_bb = self._rank_bounding_boxes(bounding_box, thresh)
        if ranked_bb is None or len(ranked_bb) <= 0: return None, intermediates

        ranked_bb.sort(key=lambda item: item[1], reverse=True)
        intermediates['ranked'] = ranked_bb

        top_bbs = [item[0] for item in ranked_bb[:k]]
        intermediates['top_bbs'] = top_bbs
        intermediates['best_bb'] = top_bbs[0]

        lps = [self._helper.crop_on_bounding_box(array, bb)
               for bb in top_bbs]
        intermediates['lps'] = lps
        intermediates['lp'] = lps[0]
        return lps, intermediates

    def _overlap(self, b1, b2):
        '''
        Returns true if 2 bounding boxes are overlapping
        '''
        tl1, br1 = b1
        tl2, br2 = b2
        if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
            return False
        if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
            return False
        return True

    def _tup(self, point):
        return (point[0], point[1])

    def _to_tl_br(self, bounding_box):
        '''
        Converts from coordinates to [top-left, bottom-right]
        '''
        x, y, w, h = bounding_box
        return [[x, y], [x + w, y + h]]

    def _to_xywh(self, tlbr):
        '''
        Converts from [top-left, bottom-right] to [x,y, w, h]
        '''
        tl, br = tlbr
        return (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])

    def _get_all_overlaps(self, boxes, bounds, index):
        '''
        Get all overlaps for a bounding box
        '''
        overlaps = []
        for a in range(len(boxes)):
            if a != index:
                if self._overlap(bounds, boxes[a]):
                    overlaps.append(a)
        return overlaps

    def _merge_candidates(self, boxes, merge_margin=15):
        '''
        Merge boxes that overlap after expanding each side by
        merge_margin pixels.
        '''

        if not boxes or len(boxes) < 2:
            return list(boxes)

        result = [self._to_tl_br(b) for b in boxes]

        finished = False
        while not finished:
            finished = True
            index = len(result) - 1
            while index >= 0:
                curr = result[index]
                expanded = [
                    [curr[0][0] - merge_margin, curr[0][1] - merge_margin],
                    [curr[1][0] + merge_margin, curr[1][1] + merge_margin],
                ]
                overlaps = self._get_all_overlaps(result, expanded, index)
                if overlaps:
                    overlaps.append(index)
                    con = []
                    for ind in overlaps:
                        tl, br = result[ind]
                        con.append([tl])
                        con.append([br])
                    x, y, w, h = cv.boundingRect(np.array(con))
                    merged = [[x, y], [x + w, y + h]]
                    for ind in sorted(overlaps, reverse=True):
                        del result[ind]
                    result.append(merged)
                    finished = False
                    break
                index -= 1

        return [self._to_xywh(b) for b in result]

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
