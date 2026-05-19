#!/usr/bin/env python3
import unittest
import os
import cv2 as cv
import random
import shutil
from src.character_segmentation import CharacterSegmentation
from src.license_plate_extraction import DetectionPipeline


class CharacterSegmentationTests(unittest.TestCase):
    _dirname = os.path.dirname(__file__)
    _data_path = os.path.abspath(os.path.join(
        _dirname, '../../data/test/'))
    _outputh_path = os.path.abspath(os.path.join(
        _dirname, './output/pipeline/segmentation'))
    _sample_size = 15
    _data = set()

    def setUp(self):
        self._get_random_files()
        self._pipeline = DetectionPipeline()
        self._segmentation = CharacterSegmentation()

    def test_segmentation(self) -> None:
        output_path = os.path.join(self._outputh_path, "lp_extraction")
        lp_path = os.path.join(output_path, "lp")

        try:
            self._create_or_clear_directory(output_path)
            self._create_or_clear_directory(lp_path)

        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return
        for file in self._data:
            try:
                image = cv.imread(os.path.join(self._data_path, file))
                segments = os.path.join(output_path, file)
                self._create_or_clear_directory(segments)
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            lp = self._pipeline.extraction_pipeline(image)
            segmented = self._segmentation.character_segmentation(lp)
            if lp is not None:
                cv.imwrite(os.path.join(lp_path, file), lp)
            if (segmented is not None):
                for segment in segmented:
                    cv.imwrite(os.path.join(segments, file), segment)
            cv.imwrite(os.path.join(output_path, file), image)

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
