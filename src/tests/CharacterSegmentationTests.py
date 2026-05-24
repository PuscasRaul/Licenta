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
        _dirname, '../../data/valid/'))
    _outputh_path = os.path.abspath(os.path.join(
        _dirname, './output/pipeline/segmentation'))
    _sample_size = 15
    _data = set()

    def setUp(self):
        self._get_random_files()
        self._pipeline = DetectionPipeline((1.5, 8.0), 500)
        self._segmentation = CharacterSegmentation()

    def test_segmentation(self) -> None:
        output_path = os.path.join(self._outputh_path,
                                   "character_segmentation")

        try:
            self._create_or_clear_directory(output_path)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            if file.endswith("xml"):
                continue
            try:
                image = cv.imread(os.path.join(self._data_path, file))
                image_path = os.path.join(output_path, file)
                self._create_or_clear_directory(image_path)
            except Exception:
                print(f'Exception occured when reading/ \
                        creating directory {file}')
                continue

            if (image is None):
                print('image was None')
                continue

            lps = self._pipeline.extraction_pipeline(image)
            if lps is None or len(lps) == 0:
                print('lp was none')
                continue

            characters = self._segmentation.character_segmentation(lps)
            if characters is not None and len(characters) > 0:
                for i, character in enumerate(characters):
                    char_filename = f"char_{i}.png"
                    full_save_path = os.path.join(image_path, char_filename)
                    cv.imwrite(full_save_path, character)
            else:
                print('characters were none')

            cv.imwrite(os.path.join(output_path, file), image)
            for i, lp in enumerate(lps):
                suffix = '' if i == 0 else f"_cand{i}"
                name, ext = os.path.splitext(file)
                cv.imwrite(os.path.join(image_path, f"{name}{suffix}{ext}"),
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


if __name__ == '__main__':
    unittest.main()
