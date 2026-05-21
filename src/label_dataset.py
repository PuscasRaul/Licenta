#!/usr/bin/env python3
import os
import cv2 as cv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, 'data', 'romanian', 'unlabled')
output_path = os.path.join(project_root, 'data', 'romanian', 'labeled')

bad_output = os.path.join(output_path, "bad")

os.makedirs(bad_output, exist_ok=True)


def create_dir(dir_name):
    dir_path = os.path.join(output_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)


def image_callback(image, key, filename):
    # for images without a character/ unrecognizable one
    if key == '.':
        cv.imwrite(os.path.join(bad_output, filename), image)
    else:
        create_dir(key)
        cv.imwrite(os.path.join(output_path, key, filename), image)


WINDOW = "label"
cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)

quit_requested = False
for (root, dirs, files) in os.walk(input_path):
    if quit_requested:
        break
    for file in files:
        image = cv.imread(os.path.join(root, file))
        if image is None:
            continue

        cv.imshow(WINDOW, image)
        key = cv.waitKey(0) & 0xFF

        # ESC or q to quit
        if key == 27 or key == ord('q'):
            quit_requested = True
            break

        # space to skip
        if key == ord(' '):
            continue

        image_callback(image, chr(key), file)

cv.destroyAllWindows()
