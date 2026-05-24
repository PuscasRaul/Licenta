#!/usr/bin/env python3
import unittest
import os
import shutil
import random
import cv2 as cv

from src.pipeline.LPExtraction import LPExtraction as LPE
from src.pipeline.HelperProcessingFunctions import HelperProcessingFunctions


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
        self._pipeline = LPE((1.5, 8.0), 500)
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

            lps, intermediates = self._pipeline.extraction_pipeline_with_intermediates(image)

            cv.imwrite(os.path.join(output_path, file), image)
            if 'thresh' in intermediates:
                cv.imwrite(os.path.join(thresholded_path, file),
                           intermediates['thresh'])
            if 'dilated' in intermediates:
                cv.imwrite(os.path.join(closing_path, file),
                           intermediates['dilated'])
            if 'top_bbs' in intermediates:
                roi = self._pipeline._lp_roi
                roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)
                annotated = image.copy()
                colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255),
                          (255, 0, 255), (255, 255, 0)]
                for i, (x, y, w, h) in enumerate(intermediates['top_bbs']):
                    color = colors[i] if i < len(colors) else (255, 0, 0)
                    cv.rectangle(annotated,
                                 (x + roi_x, y + roi_y),
                                 (x + roi_x + w, y + roi_y + h),
                                 color, 2)
                cv.imwrite(os.path.join(bb_path, file), annotated)
            if lps:
                name, ext = os.path.splitext(file)
                for i, lp in enumerate(lps):
                    suffix = '' if i == 0 else f"_{i}"
                    cv.imwrite(os.path.join(lp_path, f"{name}{suffix}{ext}"),
                               lp)

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

    def compute_roi(self, image):
        h, w = image.shape[:2]
        return (w // 4, h // 2, w // 2, h // 2)


if __name__ == '__main__':
    unittest.main()
