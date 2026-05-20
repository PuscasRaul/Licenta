import cv2 as cv
import numpy as np
from src.helpers import HelperProcessingFunctions as helper


class CharacterSegmentation():
    def __init__(self) -> None:
        return

    '''
    Implement this a tad bit better, work in a better way
    With the license plate extraction
    '''

    def character_segmentation(self, license_plate: np.array):
        if license_plate is None:
            return
        greyscale = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        thresholded = cv.adaptiveThreshold(
            greyscale,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            11,
            2)
        contours = helper.find_countours(thresholded,
                                         aspect_ratio_bounds=(0, 1))

        if contours is None:
            return

        min_height = 10
        # Aspect ratio = width / height (cnt[2] / cnt[3])
        bounding_boxes = [
            cnt for cnt in contours
            if cnt[3] > min_height and (0 < (cnt[2] / cnt[3]) <= 1)
        ]
        sorted_bb = sorted(bounding_boxes, key=lambda b: b[0])
        characters = [helper.crop_on_bounding_box(license_plate, bb)
                      for bb in sorted_bb]
        return characters
