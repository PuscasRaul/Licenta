import cv2 as cv
import numpy as np
from src.helpers import HelperProcessingFunctions as helper


class DetectionPipeline():

    def __init__(self):
        return

    def mask_center(self, array: np.array) -> np.array:
        rows, cols = np.shape(array)
        '''
        Take around 50% of the image
        50% from the top,
        25% from left and
        25% from right
        '''
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
        greyscale = cv.cvtColor(array, cv.COLOR_BGR2GRAY)
        median = helper.median_filter(array=greyscale)
        sobel = self.sobel(median)
        thresh = helper.thresholding(sobel)
        masked = self.mask_center(thresh)
        open_image = helper.opening(masked)
        dilated_image = helper.dilation(open_image)
        bounding_box = helper.find_countours(dilated_image, (2.0, 6), 1000)
        lp = helper.crop_on_bounding_box(array, bounding_box)
        return lp
