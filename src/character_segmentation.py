import cv2 as cv
import numpy as np
from src.helpers import HelperProcessingFunctions as helper


class CharacterSegmentation():
    def __init__(self) -> None:
        return

    def character_segmentation(self, license_plate: np.array):
        if license_plate is None:
            return
        greyscale = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        thresholded = helper.otsu_thresholding(greyscale)
        contours = helper.find_countours(thresholded,
                                         aspect_ratio_bounds=(0.15, 1))

        if contours is None:
            return

        plate_height = thresholded.shape[0]
        contours = [box for box in contours
                    if 0.4 <= box[3] / plate_height <= 0.95]
        sorted_cnt = sorted(contours, key=lambda b: b[0])  # sort on x-axis
        return [helper.crop_on_bounding_box(license_plate, cnt)
                for cnt in sorted_cnt]
