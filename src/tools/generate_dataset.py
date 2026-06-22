#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
from src.pipeline.LPExtraction import LPExtraction as DP
from src.pipeline.CharacterSegmentation import CharacterSegmentation as CS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, 'data')
output_path = os.path.join(input_path, 'unlabled')
lp_path = os.path.join(input_path, "lp")
splits = ["valid"]

os.makedirs(output_path, exist_ok=True)
os.makedirs(lp_path, exist_ok=True)

extraction_pipeline = DP((1.5, 8.0), 500)
segmentation_pipeline = CS()


for split in splits:
    direc = os.path.join(input_path, split)
    for (root, dirs, files) in os.walk(direc):
        for file in files:
            if file.endswith("xml"):
                continue

            image = cv.imread(os.path.join(root, file))
            if image is None:
                continue

            lps = extraction_pipeline.extraction_pipeline(image)
            if lps is None or len(lps) == 0:
                continue

            characters = segmentation_pipeline.character_segmentation(lps)
            if characters is None or len(characters) <= 0:
                continue

            name, _ = os.path.splitext(file)
            lp_out = os.path.join(lp_path, file)
            cv.imwrite(lp_out, lps[0])
            for idx, character in enumerate(characters):
                file_out = os.path.join(output_path, f"{name}_{idx}.png")
                cv.imwrite(file_out, character)
