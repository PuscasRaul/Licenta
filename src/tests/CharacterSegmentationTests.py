#!/usr/bin/env python3
import unittest
import os
import cv2 as cv
import numpy as np
import random
import shutil
from src.pipeline.CharacterSegmentation import CharacterSegmentation as CS
from src.pipeline.LPExtraction import LPExtraction as LPE


class CharacterSegmentationTests(unittest.TestCase):
    _dirname = os.path.dirname(__file__)
    _data_path = os.path.abspath(os.path.join(
        _dirname, '../../data/VALIDATE/'))
    _outputh_path = os.path.abspath(os.path.join(
        _dirname, './output/pipeline/segmentation'))
    _sample_size = 15
    _data = set()

    def setUp(self):
        self._get_random_files()
        self._pipeline = LPE((1.5, 8.0), 500)
        self._segmentation = CS()

    def test_segmentation(self) -> None:
        '''
        Visualization harness for tuning character segmentation.

        For every sampled image it writes, per LP candidate:
          - <name>_cand<i>.png            the raw LP crop
          - <name>_cand<i>_boxes.png      LP with character bounding boxes
          - <name>_cand<i>_binary.png     the binarized image + text band
          - chars/cand<i>_char_<k>.png    each cropped character
        The candidate folders are ordered best-first by the segmentation
        score so the chosen result is cand0.
        '''
        output_path = os.path.join(self._outputh_path,
                                   "character_segmentation")
        try:
            self._create_or_clear_directory(output_path)
        except Exception as e:
            print(f'Exception occured when creating file {e}')
            return

        total_chars = 0
        for file in self._data:
            if file.endswith("xml"):
                continue
            try:
                image = cv.imread(os.path.join(self._data_path, file))
                name, _ = os.path.splitext(file)
                image_path = os.path.join(output_path, name)
                self._create_or_clear_directory(image_path)
            except Exception:
                print(f'Exception occured when reading/ \
                        creating directory {file}')
                continue

            if image is None:
                print(f'image was None: {file}')
                continue

            lps = self._pipeline.extraction_pipeline(image)
            if lps is None or len(lps) == 0:
                print(f'no lp candidates: {file}')
                continue

            results = self._segmentation.segment_all(lps)
            if not results:
                print(f'segmentation produced nothing: {file}')
                continue

            chars_dir = os.path.join(image_path, 'chars')
            self._create_or_clear_directory(chars_dir)

            for i, res in enumerate(results):
                lp = res['lp']
                bboxes = res['bboxes']
                cv.imwrite(os.path.join(image_path, f"cand{i}.png"), lp)
                cv.imwrite(os.path.join(image_path, f"cand{i}_boxes.png"),
                           self._annotate(lp, bboxes))
                cv.imwrite(os.path.join(image_path, f"cand{i}_binary.png"),
                           self._binary_view(res['debug']))

                for k, character in enumerate(res['chars']):
                    if character is None or character.size == 0:
                        continue
                    cv.imwrite(
                        os.path.join(chars_dir, f"cand{i}_char_{k}.png"),
                        character)

                print(f"{file} cand{i}: {len(bboxes)} chars "
                      f"score={res['score']:.3f}")

            total_chars += len(results[0]['chars'])

        print(f"\nTotal characters from best candidates: {total_chars}")

    @staticmethod
    def _annotate(lp, bboxes):
        vis = (lp.copy() if lp.ndim == 3
               else cv.cvtColor(lp, cv.COLOR_GRAY2BGR))
        for (x, y, w, h) in bboxes:
            cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return vis

    @staticmethod
    def _binary_view(debug):
        bw = debug.get('bw') if debug else None
        if bw is None:
            return np.zeros((10, 10), np.uint8)
        vis = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)
        band = debug.get('band')
        if band is not None:
            y0, y1 = band
            cv.line(vis, (0, y0), (vis.shape[1] - 1, y0), (0, 0, 255), 1)
            cv.line(vis, (0, y1), (vis.shape[1] - 1, y1), (0, 0, 255), 1)
        return vis

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
