#!/usr/bin/env python3
import unittest
import os
import shutil
import random
import cv2 as cv
import numpy as np
from src.license_plate_extraction import DetectionPipeline
from src.helpers import HelperProcessingFunctions


class DetectionPipelineTests(unittest.TestCase):
    _dirname = os.path.dirname(__file__)
    _data_path = os.path.abspath(os.path.join(
        _dirname, '../../data/test/'))
    _outputh_path = os.path.abspath(os.path.join(
        _dirname, './output/pipeline/'))
    _sample_size = 15
    _data = set()

    def setUp(self):
        self._get_random_files()
        self._pipeline = DetectionPipeline((2.0, 5), 1500)
        self._helper = HelperProcessingFunctions()

    def test_pipeline(self) -> None:
        output_path = os.path.join(self._outputh_path, "lp_extraction")
        thresholded_path = os.path.join(output_path, "thresholded")
        bb_path = os.path.join(output_path, "bb")
        closing_path = os.path.join(output_path, "closing")
        lp_path = os.path.join(output_path, "lp")

        try:
            self._create_or_clear_directory(output_path)
            self._create_or_clear_directory(thresholded_path)
            self._create_or_clear_directory(bb_path)
            self._create_or_clear_directory(closing_path)
            self._create_or_clear_directory(lp_path)

        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return
        for file in self._data:
            try:
                image = cv.imread(os.path.join(self._data_path, file))
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            if image is None:
                continue

            greyscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            median = self._helper.median_filter(greyscale)
            sobel = self._pipeline.sobel(median)
            otsu = self._helper.otsu_thresholding(sobel)
            otsu = cv.bitwise_not(otsu)
            opening = self._helper.opening(otsu)
            dilation = self._helper.dilation(opening)
            bb = self._helper.find_countours(dilation, (2.0, 5), 1500)
            cv.imwrite(os.path.join(output_path, file), image)
            cv.imwrite(os.path.join(thresholded_path, file), otsu)
            cv.imwrite(os.path.join(closing_path, file), dilation)
            if bb is not None:
                ranked_bb = self._rank_bounding_boxes(bb, otsu)
                if ranked_bb is None or len(ranked_bb) <= 0:
                    return None

                ranked_bb.sort(key=lambda item: item[1], reverse=True)
                best_bb = ranked_bb[0][0]
                (x, y, w, h) = best_bb
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.imwrite(os.path.join(bb_path, file), image)

    def _rank_bounding_boxes(self, bb, otsu):
        if bb is None:
            return None

        scored_boxes = []
        for (x, y, w, h) in bb:
            area = w * h

            roi_edges = otsu[y:y+h, x:x+w]
            edge_density = cv.countNonZero(roi_edges) / area
            score = area * edge_density
            scored_boxes.append(((x, y, w, h), score))

        return scored_boxes

    def _get_random_files(self) -> None:
        files = []
        for root, dirs, filenames in os.walk(self._data_path):
            for filename in filenames:
                files.append(filename)

        if len(files) == 0:
            return

        actual_size = min(self._sample_size, len(files))
        self._data = set(random.sample(files, actual_size))

    def _create_or_clear_directory(self, path):
        try:
            os.makedirs(path)
        except FileExistsError:
            print(f'Directory found at {path} deleting its contents')
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
        except PermissionError:
            print(f"Permission denied: Unable to create '{path}'.")
            raise PermissionError
        except Exception as e:
            print(f"An error occurred: {e}")
            raise Exception


if __name__ == '__main__':
    unittest.main()
