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

    @staticmethod
    def VEDA(array: np.array) -> np.array:
        '''
        Perform VEDA on a 2D binarized array
        :param array: 2D binarized array

        :return: 2D binarized array after VEDA has been applied
        '''

        # TODO: Finish this function tomorrow
        veda_array = np.zeros_like(array)
        it = np.nditer(array, flags=['multi_index'])
        for coord in it:

            row = coord[0]
            col = coord[1]
            left_col = coord[1] - 1
            right_col = coord[1] + 1
            right_right_col = coord[1] + 2
            down_row = coord[0] + 1

            if left_col < 0:
                left_col = 0
            if right_col >= np.shape(array)[1]:
                right_col = col
            if right_right_col >= np.shape(array)[1]:
                right_right_col = right_col
            if down_row >= np.shape(array)[0]:
                down_row = row

            center_cond = array[row][col] == 0 \
                and array[row][right_col] == 0\
                and array[down_row][col] == 0 \
                and array[down_row][right_col] == 0
            left_cond = array[row][left_col] == 0 \
                and array[down_row][left_col] == 0
            right_cond = array[down_row][right_right_col] == 0 \
                and array[down_row][right_right_col] == 0

            if not center_cond and not left_cond and not right_cond:
                veda_array[coord[0]][coord[1]] = 255
                veda_array[coord[0]][coord[1]+1] = 255

        return veda_array
