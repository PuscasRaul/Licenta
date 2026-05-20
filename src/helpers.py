import cv2 as cv
import numpy as np


class HelperProcessingFunctions():

    def __init__(self):
        return

    @staticmethod
    def binary_array(array: np.array, thresh, value=0) -> np.array:
        '''
        Return a 2D binary array
        :param array: np.2D array
        :param thresh: Value used for thresholding
        :param value: Output value when between threshold

        :return: Binary 2D array
        '''
        high_value = 255
        if value == 0:
            binary = np.full_like(array, high_value)
        else:
            binary = np.zeros_like(array)
            value = 1

        '''
        if value == 0, make all values in binary 0 if the
        corresponding value in the input array is between the
        threshold
        Otherwise, the value remains as 1.
        '''
        binary[(array >= thresh[0]) & (array <= thresh[1])] = value
        return binary

    @staticmethod
    def thresholding(array: np.array) -> np.array:
        thresh = [0, cv.mean(array)[0] * 2]
        return HelperProcessingFunctions.binary_array(array, thresh)

    @staticmethod
    def otsu_thresholding(array: np.array) -> np.array:
        '''
        Binarize a grayscale image with Otsu's method.
        Inverted (THRESH_BINARY_INV) so dark characters become
        white foreground, as cv.findContours detects white blobs.
        :param array: np 2D grayscale array
        :return: Binary 2D array
        '''
        _, binary = cv.threshold(
            array, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return binary

    @staticmethod
    def median_filter(array: np.array, ksize=3) -> np.array:
        return cv.medianBlur(array, ksize)

    @staticmethod
    def opening(array, ksize=(3, 3)) -> np.array:
        kernel = np.ones(ksize, np.uint8)
        return cv.morphologyEx(array, cv.MORPH_OPEN, kernel)

    @staticmethod
    def dilation(array: np.array) -> np.array:
        kernel = np.ones((1, 8), np.uint8)
        return cv.dilate(array, kernel, iterations=3)

    @staticmethod
    def find_countours(array, aspect_ratio_bounds=None, size_constraint=None):
        if array is None:
            return None

        contours, _ = cv.findContours(
            array,
            cv.RETR_LIST,
            cv.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if h == 0:
                continue

            area = h * w
            aspect_ratio = float(w) / h
            aspect_ratio_cond = (aspect_ratio_bounds is None or
                                 (aspect_ratio_bounds[0] <= aspect_ratio <=
                                  aspect_ratio_bounds[1]))
            size_constraint_cond = (size_constraint is None or
                                    area >= size_constraint)

            if aspect_ratio_cond and size_constraint_cond:
                boxes.append(cnt)

        if len(boxes) > 0:
            return [cv.boundingRect(cnt) for cnt in boxes]
        return None

    @staticmethod
    def crop_on_bounding_box(array, bounding_box):
        (x, y, w, h) = bounding_box
        return array[y:y+h, x:x+w]
