#!/usr/bin/env python3
import unittest
import os
import shutil
import random
import cv2 as cv
from src.filters import DetectionPipeline


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
        self._pipeline = DetectionPipeline()

    def test_binary_array(self) -> None:
        output_path = os.path.join(self._outputh_path, "binary_array/")
        try:
            self._create_or_clear_directory(output_path)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            try:
                image = cv.imread(
                    os.path.join(self._data_path, file),
                    cv.IMREAD_GRAYSCALE
                )
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            thresh = self._pipeline.adaptive_thresholding(array=image)
            cv.imwrite(os.path.join(output_path, file), thresh)

    def test_ulea(self) -> None:
        output_path = os.path.join(self._outputh_path, "ulea/")
        try:
            self._create_or_clear_directory(output_path)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            try:
                image = cv.imread(
                    os.path.join(self._data_path, file),
                    cv.IMREAD_GRAYSCALE
                )
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            thresh = self._pipeline.adaptive_thresholding(array=image)
            ulea = self._pipeline.ulea(thresh)
            cv.imwrite(os.path.join(output_path, file), ulea)

    def test_veda(self) -> None:
        output_path = os.path.join(self._outputh_path, "veda/")
        output_path_binary = os.path.join(output_path, "binary/")
        output_path_ulea = os.path.join(output_path, "ulea/")
        try:
            self._create_or_clear_directory(output_path)
            self._create_or_clear_directory(output_path_binary)
            self._create_or_clear_directory(output_path_ulea)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            try:
                image = cv.imread(
                    os.path.join(self._data_path, file),
                    cv.IMREAD_GRAYSCALE
                )
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            thresh = self._pipeline.binary_array(image, [128, 255])
            ulea = self._pipeline.ulea(thresh)
            veda = self._pipeline.veda(ulea)
            cv.imwrite(os.path.join(output_path_binary, file), thresh)
            cv.imwrite(os.path.join(output_path_ulea, file), ulea)
            cv.imwrite(os.path.join(output_path, file), veda)

    def test_skip_image(self) -> None:
        output_path = os.path.join(self._outputh_path, "skip_image/")
        output_veda = os.path.join(output_path, "veda")
        try:
            self._create_or_clear_directory(output_path)
            self._create_or_clear_directory(output_veda)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            try:
                image = cv.imread(
                    os.path.join(self._data_path, file),
                    cv.IMREAD_GRAYSCALE
                )
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            thresh = self._pipeline.binary_array(image, [0, 128])
            ulea = self._pipeline.ulea(thresh)
            veda = self._pipeline.veda(ulea)
            skip_image = self._pipeline.skip_image(veda)
            cv.imwrite(os.path.join(output_path, file), skip_image)
            cv.imwrite(os.path.join(output_veda, file), veda)
        return

    '''
    def test_integral_image(self) -> None:
        output_path = os.path.join(self._outputh_path, "integral_image/")
        try:
            self._create_or_clear_directory(output_path)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        for file in self._data:
            try:
                image = cv.imread(
                    os.path.join(self._data_path, file),
                    cv.IMREAD_GRAYSCALE
                )
            except Exception:
                print(f'Exception occured when reading {file}')
                continue
            thresh = self._pipeline.adaptive_thresholding(image)
            ulea = self._pipeline.ulea(thresh)
            veda = self._pipeline.veda(ulea)
            skip_image = self._pipeline.skip_quantity(veda, np.shape(veda)[1])
            integral_image = self._pipeline.integral_image(skip_image)
            cv.imwrite(os.path.join(output_path, file), integral_image)
    '''

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
