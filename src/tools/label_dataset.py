#!/usr/bin/env python3
import os
import cv2 as cv
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, 'data', 'unlabled')
output_path = os.path.join(project_root, 'data', 'labeled')

bad_output = os.path.join(output_path, "bad")

os.makedirs(bad_output, exist_ok=True)


def image_callback(image, key, filename):
    # for images without a character/ unrecognizable one
    initial_location = os.path.join(input_path, filename)

    if key == '.':
        final_location = os.path.join(bad_output, filename)
        shutil.move(initial_location, final_location)

    else:
        dir_path = os.path.join(output_path, key)
        os.makedirs(dir_path, exist_ok=True)
        final_location = os.path.join(dir_path, filename)
        shutil.move(initial_location, final_location)


WINDOW = "label"
cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)

quit_requested = False
for (root, dirs, files) in os.walk(input_path):
    if quit_requested:
        break
    for file in files:
        filename = os.path.join(root, file)
        image = cv.imread(filename)
        if image is None:
            continue

        cv.imshow(WINDOW, image)
        key = cv.waitKey(0) & 0xFF

        # ESC or q to quit
        if key == 27 or key == ord('\\'):
            quit_requested = True
            break

        # space to skip
        if key == ord(' '):
            continue

        image_callback(image, chr(key), file)

cv.destroyAllWindows()
