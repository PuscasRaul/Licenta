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

    @staticmethod
    def integral_image(array: np.array) -> np.array:
        '''
        integral image is an algorithm for generating the sum of
        values in a rectangular subset of a grid quickly and efficiently
        :param array: Binarized 2D array, after VEDA or Sobel applied
        :param value_array: Output array of same shape as array
        :return :2D array
        TODO: Find a better documentation
        '''
        it = np.nditer(array, flags=["multi_index"])
        result = np.zeros_like(array)
        for coord in it:
            x = coord[0]
            y = coord[1]
            if x == 0:
                sat = 0
            else:
                sat = result[x-1][y]
            if y == 0:
                s = 0
            else:
                s = result[x][y-1]
            result[x][y] = sat + s + array[x][y]

        return result

    @staticmethod
    def skip_quantity(array: np.array, image_width) -> np.array:
        '''
        skip quantity is the skip number from white pixel to black pixel or
        from black pixel to white pixel in the rectangl
        :param array: binarized array after Sobel/VEDA has been applied
        :return :Not yet defined

        S(x, y) = S'(x + w/2, y) - S'(x - w/2, y)
        S'(x, y) = S'(x - 1, y) + E(x-1, y)(x, y)

        E(x-1, y)(x, y) = 1, BE(x-1, y) != BE(x, y)
                          0, otherwise
        '''

        image_half = image_width // 2
        transitions = np.zeros_like(array)
        transitions[1:, :] = (array[1:, :] != array[:-1, :]).astype(int)
        s_prime = np.cumsum(transitions, axis=0)
        rows, cols = array.shape
        skipped_image = np.zeros_like(array)
        idx_x = np.arange(rows)
        idx_right = np.clip(idx_x + image_half, 0, rows - 1)
        idx_left = np.clip(idx_x - image_half, 0, rows - 1)
        skipped_image = s_prime[idx_right, :] - s_prime[idx_left, :]
        return skipped_image

    @staticmethod
    def connected_component_analysis() -> np.array:

        return None
