#!/usr/bin/env python3
import cv2 as cv
import json
import os
from pathlib import Path

clicked_points = []  # clicked points array for cv.rectangle coordinates


def tlbr_xywh(coordinates):
    return [coordinates[0][0],
            coordinates[0][1],
            coordinates[1][0] - coordinates[0][0],
            coordinates[1][1] - coordinates[0][1]
            ]


def mouse_callback(event, x, y, flags, param) -> None:
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param) < 2:
            print(x, y)
            param.append((x, y))


def generate_xml(coordinates, filename) -> None:
    filename_only = os.path.basename(filename)  # strip extension

    json_file = Path(filename).with_suffix('.json')
    json_data = {"filename": filename_only, "bbox": coordinates}

    with open(json_file, 'w') as f:
        f.write(json.dumps(json_data))


width = 540
height = 540

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', width, height)
cv.setMouseCallback('image', mouse_callback, clicked_points)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath(os.path.join(project_root, "data"))
img_dir = os.path.join(data_path, "dataset_romania")

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for root, subdirs, files in os.walk(img_dir):
    for subdir in subdirs:
        print(f'subdirectory: {subdir}')

    for filename in files:
        file_path = os.path.join(root, filename)
        image = cv.imread(file_path)
        clicked_points.clear()
        display_image = image.copy()  # just for display purposes

        while len(clicked_points) < 2:
            for point in clicked_points:
                cv.circle(display_image, point, 5, (0, 0, 255), -1)
            cv.imshow('image', display_image)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                cv.destroyAllWindows()
                exit()

        p1, p2 = clicked_points
        cv.rectangle(display_image, p1, p2, (0, 255, 0), 3)
        cv.imshow('image', display_image)
        coordinates = tlbr_xywh([p1, p2])
        generate_xml(coordinates, file_path)
        cv.waitKey(5000)
