import cv2 as cv
import numpy as np
from src.helpers import HelperProcessingFunctions as helper


class DetectionPipeline():

    def __init__(self, lp_aspect_ratio, lp_min_size):
        '''
        :param lp_aspect_ratio :tuple aspect ratio for a contour to be
        considered a license plate
        :param lp_min_size minimum area size of a region to be considered
        a candidate for lp
        '''
        self._lp_aspect_ratio = lp_aspect_ratio
        self._lp_min_size = lp_min_size

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

    def extraction_pipeline(self, array: np.array):
        if array is None:
            return None

        greyscale = cv.cvtColor(array, cv.COLOR_BGR2GRAY)
        median = helper.median_filter(array=greyscale)
        sobel = self.sobel(median)
        thresh = helper.otsu_thresholding(sobel)
        thresh = cv.bitwise_not(thresh)
        open_image = helper.opening(thresh)
        dilated_image = helper.dilation(open_image)
        bounding_box = helper.find_countours(dilated_image,
                                             self._lp_aspect_ratio,
                                             self._lp_min_size)
        if bounding_box is None or len(bounding_box) <= 0:
            return None

        ranked_bb = self._rank_bounding_boxes(bounding_box, thresh)
        if ranked_bb is None or len(ranked_bb) <= 0:
            return None

        ranked_bb.sort(key=lambda item: item[1], reverse=True)
        best_bb = ranked_bb[0][0]

        lp = helper.crop_on_bounding_box(array, best_bb)
        return lp

    def _rank_bounding_boxes(self, bb, otsu):
        if bb is None:
            return None

        scored_boxes = []
        for (x, y, w, h) in bb:
            area = w * h

            roi_edges = otsu[y:y+h, x:x+w]
            edge_density = cv.countNonZero(roi_edges) / area
            score = area * edge_density
            scored_boxes.append(((x, y, w, h), score))

        return scored_boxes
