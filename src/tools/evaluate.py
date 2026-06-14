#!/usr/bin/env python3
'''
End-to-end run + stage-level evaluation of the IORA pipeline.
'''
import json
import os
import shutil
import cv2 as cv
from src.pipeline.ProcessingPipeline import ProcessingPipeline
from src.pipeline.CharacterRecognition import CharacterRecognition
from src.pipeline.CharacterSegmentation import CharacterSegmentation
from src.pipeline.LPExtraction import LPExtraction
from src.tools.metrics import load_ground_truth, best_iou

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath(os.path.join(project_root, "data"))
model_path = os.path.abspath(os.path.join(data_path, "svm_model.joblib"))

loc = LPExtraction((1.5, 8.0), 300)
seg = CharacterSegmentation()
rec = CharacterRecognition(model_path=model_path)
pipeline = ProcessingPipeline(loc, seg, rec)


def validate(validation_dir, output_dir, ground_truth=None):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for root, dirs, filenames in os.walk(validation_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in image_exts:
                continue
            output_path = os.path.join(output_dir, filename)
            character_directory = os.path.join(output_path,
                                               "characters")
            recognition_directory = os.path.join(output_path,
                                                 'recognition')

            print(f'Processing {filename}...')
            _create_or_clear_directory(output_path)

            image = cv.imread(os.path.join(root, filename))
            if image is None:
                print(f'Image could not be read for {filename}')
                continue
            license_plate, intermediates = \
                loc.extraction_pipeline_with_intermediates(image, top_k=5)
            top_bbs = intermediates.get('top_bbs', [])

            with open(os.path.join(output_path, 'bboxes.json'), 'w') as f:
                json.dump([list(bb) for bb in top_bbs], f)

            if ground_truth is not None:
                gt_boxes = ground_truth.get(filename, [])
                if gt_boxes is None or len(gt_boxes) <= 0:
                    print(f'No ground truth for {filename}')
                score = best_iou(top_bbs, gt_boxes)
                with open(os.path.join(output_path, 'iou.txt'), 'w') as f:
                    f.write(f'{score:.4f}')

            if license_plate is None or len(license_plate) <= 0:
                with open("lp.txt", "w", encoding="utf-8") as f:
                    f.write("No license plate found in image")
                continue

            index = 0
            for lp in license_plate:
                cv.imwrite(os.path.join(output_path, f'{filename}_lp_{index}.jpg'),
                           lp)
                index += 1

            characters = seg.character_segmentation(license_plate)

            _create_or_clear_directory(character_directory)
            if not characters:
                with open(os.path.join(character_directory, 'None'),
                          "w", encoding="utf-8") as f:
                    f.write("No characters found")
                continue
            else:
                for index, char in enumerate(characters):
                    cv.imwrite(
                        os.path.join(character_directory, f'char_{index}.jpg'),
                        char
                    )

            _create_or_clear_directory(recognition_directory)
            prediction = rec.predict(characters)
            with open(os.path.join(recognition_directory, 'Rec'),
                      "w", encoding="utf-8") as f:
                if prediction is None:
                    f.write("No characters found")
                else:
                    f.write(f'{prediction}')


def _create_or_clear_directory(path):
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


print("Please insert the validation directory, relative to the project root_directory")
validation_dir_rel = input()
validation_dir_abs = os.path.join(project_root, validation_dir_rel)
print("Please insert the output directory, relative to the project's root directory")
output_path = input()
output_abs = os.path.join(project_root, output_path)
gt = load_ground_truth(validation_dir_abs)
validate(validation_dir_abs, output_abs, ground_truth=gt)
