import cv2 as cv
import numpy as np


class CharacterSegmentation():
    def __init__(self) -> None:
        return

    def find_countours(self, array: np.array):
        contours, _ = cv.findContours(
            array,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        max_area = -1
        character_contours = []

        '''
        This should be the main algorithm, the problem now becomes
        How to select the specific aspect ratio required for
        Selecting a character from the cropped license plate,
        So that if any other object does get found, it falls
        '''
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if h == 0:
                continue

            area = h * w
            aspect_ratio = float(w) / h
            is_plate_shape = 2.0 <= aspect_ratio <= 6
            is_big_enough = area >= 1000

            if is_plate_shape and is_big_enough:
                character_contours.append(cnt)

        if len(character_contours) <= 0:
            return None

        bounding_boxes = [cv.boundingRect(cnt) for cnt in character_contours]
        return bounding_boxes.sort()
