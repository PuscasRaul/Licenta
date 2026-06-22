import cv2 as cv
import os
import xml.etree.ElementTree as ET


def parse_XML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    character_bb = []

    for item in root.findall('./object'):
        character = item.find('name').text
        bnd = item.find('bndbox')
        xmin = bnd.find('xmin').text
        ymin = bnd.find('ymin').text
        xmax = bnd.find('xmax').text
        ymax = bnd.find('ymax').text
        bndbox = (xmin, ymin, xmax, ymax)
        character_bb.append((character, bndbox))

    return character_bb


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.join(project_root, 'data', 'LP-characters')
annotations = os.path.join(root_path, "annotations")
images = os.path.join(root_path, "images")

output_path = os.path.join(root_path, "labeled")
os.makedirs(output_path, exist_ok=True)

for (dirpath, dirs, files) in os.walk(images):
    for file in files:
        image = cv.imread(os.path.join(dirpath, file))
        if image is None:
            print(f'Could not read image {file}')
            continue

        file_without_extenstion = os.path.splitext(file)[0]
        annotation = os.path.join(annotations,
                                  f'{file_without_extenstion}.xml')

        annotated = parse_XML(annotation)
        for bbox in annotated:
            character = bbox[0]
            (x, y, xmax, ymax) = map(int, bbox[1])
            crop = image[y:ymax, x:xmax]

            output_dir = os.path.join(output_path, character)
            os.makedirs(output_dir, exist_ok=True)
            cv.imwrite(os.path.join(output_dir, file), crop)
