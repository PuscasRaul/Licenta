import cv2 as cv
import numpy as np

'''
TODO: Write the code in a more "Pythonic" way
'''


class DetectionPipeline():

    def __init__(self):
        return

    def binary_array(self, array: np.array, thresh, value=0) -> np.array:
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

    def adaptive_thresholding(self, array: np.array) -> np.array:
        thresh_mean = cv.adaptiveThreshold(
            array,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            199,
            5
        )
        return thresh_mean

    def ulea(self, array: np.array) -> np.array:
        '''
        Perform ULEA on a 2D binarized array based on the provided pseudocode.
        Assumes: 0 is black (target), 255 is white (background/neighbors).
        '''
        if array.ndim != 2:
            raise ValueError("Incorrect array shape for ULEA")
        output = np.copy(array)
        max_rows, max_cols = np.shape(array)

        for r in range(1, max_rows - 1):
            for c in range(1, max_cols - 1):
                if array[r, c] == 0:
                    horiz = (array[r, c-1] == 255 and array[r, c+1] == 255)
                    vert = (array[r-1, c] == 255 and array[r+1, c] == 255)
                    diag1 = (array[r+1, c-1] == 255 and array[r-1, c+1] == 255)
                    diag2 = (array[r-1, c-1] == 255 and array[r+1, c+1] == 255)

                    if horiz or vert or diag1 or diag2:
                        output[r, c] = 255

        return output

    def veda(self, array: np.array) -> np.array:
        '''
        Perform VEDA on a 2D binarized array
        :param array: 2D binarized array

        :return: 2D binarized array after VEDA has been applied
        '''

        veda_array = np.full_like(array, 255)
        it = np.nditer(array, flags=['multi_index'])
        for coord in it:
            row, col = it.multi_index
            left_col = col - 1
            right_col = col + 1
            right_right_col = col + 2
            down_row = row + 1

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
            right_cond = array[row][right_right_col] == 0 \
                and array[down_row][right_right_col] == 0

            if center_cond and (not left_cond or not right_cond):
                veda_array[row][col] = 0
                if row + 1 >= np.shape(veda_array)[0]:
                    veda_array[row][col] = 0
                else:
                    veda_array[row + 1][col] = 0

        return veda_array

    def integral_image(self, array: np.array) -> np.array:
        '''
        integral image is an algorithm for generating the sum of
        values in a rectangular subset of a grid quickly and efficiently
        :param array: Binarized 2D array, after VEDA or Sobel applied
        :param value_array: Output array of same shape as array
        :return :2D array
        '''
        it = np.nditer(array, flags=["multi_index"])
        result = np.zeros_like(array)

        for coord in it:
            x, y = it.multi_index
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

    def skip_image(self, array: np.array) -> np.array:
        '''
        skip quantity is the skip number from white pixel to black pixel or
        from black pixel to white pixel in the rectangl
        :param array: binarized array after VEDA has been applied
        :return :Not yet defined

        S(x, y) = S'(x + w/2, y) - S'(x - w/2, y)
        S'(x, y) = S'(x - 1, y) + E(x-1, y)(x, y)

        E(x-1, y)(x, y) = white, BE(x-1, y) != BE(x, y)
                          black, otherwise
        '''
        half_width = np.shape(array)[0]
        delta = np.zeros_like(array)
        rows, cols = np.shape(array)
        iter = np.nditer(array, flags=['multi_index'])
        for _ in iter:
            row, col = iter.multi_index
            prev_row = max(0, row - 1)
            element = array[row][col]
            prev_element = array[prev_row][col]

            if element != prev_element:
                delta[row][col] = 1
            else:
                delta[row][col] = 0

        np.cumsum(a=delta, axis=0, out=delta)
        skipped_image = np.zeros_like(array)
        for _ in iter:
            row, col = iter.multi_index

            up = array[row - half_width][col]
            down = array[row + half_width][col]
            skipped_image[row][col] = up + down

        return skipped_image



    def connected_component_analysis(self, array: np.array) -> np.array:
        '''
        Connected component analysis represents an algorithm useful
        for character segmentation.
        It represents the second stage of the pipeline. Before the SVM
        gets to run on them, and after we have thresholded and located
        the license plate.

        TODO: check how to sort these in order from left to right
        '''
        output = np.zeros_like(array.shape)
        analysis = cv.connectedComponentsWithStats(array, 4, cv.CV_32S)
        (total_labels, label_ids, values, centroid) = analysis
        for i in range(1, total_labels):
            area = values[i, cv.CC_STAT_AREA]

            if (area > 140 and (area < 400)):
                component_mask = (label_ids == i).astype("uint8") * 255
                output = cv.bitwise_or(output, component_mask)

        return output
