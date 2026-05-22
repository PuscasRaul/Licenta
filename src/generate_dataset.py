#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
from src.license_plate_extraction import DetectionPipeline as DP
from src.character_segmentation import CharacterSegmentation as CS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, 'data')
output_path = os.path.join(input_path, 'unlabled')
lp_path = os.path.join(input_path, "lp")
splits = ["valid"]

os.makedirs(output_path, exist_ok=True)
os.makedirs(lp_path, exist_ok=True)

extraction_pipeline = DP((3.0, 6.0), 2000)
segmentation_pipeline = CS()


def compute_roi(image):
    h, w = image.shape[:2]
    return (w // 4, h // 2, w // 2, h // 2)


for split in splits:
    direc = os.path.join(input_path, split)
    for (root, dirs, files) in os.walk(direc):
        for file in files:
            if file.endswith("xml"):
                continue

            image = cv.imread(os.path.join(root, file))
            if image is None:
                continue

            extraction_pipeline._lp_roi = compute_roi(image)
            lp = extraction_pipeline.extraction_pipeline(image)
            if lp is None:
                continue

            characters = segmentation_pipeline.character_segmentation(lp)
            if characters is None or len(characters) <= 0:
                continue

            name, _ = os.path.splitext(file)
            lp_out = os.path.join(lp_path, file)
            cv.imwrite(lp_out, lp)
            for idx, character in enumerate(characters):
                file_out = os.path.join(output_path, f"{name}_{idx}.png")
                cv.imwrite(file_out, character)
