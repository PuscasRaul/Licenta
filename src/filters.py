import cv2 as cv
import numpy as np


class ProcessingFunctions():
    @staticmethod
    def binary_array(array: np.array, thresh, value=0) -> np.array:
        '''
        Return a 2D binary array
        :param array: np.2D array
        :param thresh: Value used for thresholding
        :param value: Output value when between threshold

        :return: Binary 2D array
        '''
        if value == 0:
            binary = np.ones_like(array)
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
    def ULEA(array: np.array, ul_width=1) -> np.array:
        '''
        Perform ULEA on a 2D binarized array
        :param array: np.2D array
        :param ul_width: unwanted line width

        :return: 2D binarized array with unwanted lines removed
        '''

        if array.dim != 2:
            raise ValueError("Incorrect array shape for ULEA")

        kernel_height = ul_width + 1
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_height))
        cleaned_array = cv.morphologyEx(array, cv.MORPH_OPEN, kernel)
        return cleaned_array
