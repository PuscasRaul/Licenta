import cv2 as cv
import numpy as np


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
        grad_y = cv.Sobel(array, ddepth, 0, 1, ksize=3, scale=scale,
                          delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

    def grad4_thresholding(self, array: np.array) -> np.array:
        thresh = [0, cv.mean(array)[0] * 6]
        return self.binary_array(array, thresh)

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

    def skip_image(self, array: np.ndarray, w: int) -> np.ndarray:
        """
        Calculates the skip image based on horizontal edge transitions.

        :param array: binarized image array (2D numpy array)
        :param w: width of the sliding window for equation (6)
        :return: 2D numpy array containing S(x, y) skip quantities
        """
        rows, cols = array.shape
        half_w = w // 2

        transitions = np.diff(array, axis=1) != 0
        delta = np.zeros((rows, cols), dtype=np.int32)

        delta[:, 1:] = transitions.astype(np.int32) * 255
        S_prime = np.cumsum(delta, axis=1)
        S = np.zeros((rows, cols), dtype=np.int32)
        for x in range(cols):
            x_plus = min(x + half_w, cols - 1)
            x_minus = max(x - half_w, 0)
            S[:, x] = S_prime[:, x_plus] - S_prime[:, x_minus]

        return S

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
