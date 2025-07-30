#based off of the implementation from https://medium.com/@Spritan/convert-yolo-annotations-to-voc-ee5745b05851 

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

class_names = ["NA", "fire", "smoke"]
custom_labels = ["fire", "smoke"]

REMOVE_EMPTY_ANNOTATIONS = True

#paths
yolo_dir_path = os.path.expanduser("~/jetson-inference/python/training/detection/ssd/data/fire-and-smoke-dataset-object-detection-yolo/")
voc_dir_path  = os.path.expanduser("~/jetson-inference/python/training/detection/ssd/data/fire_smoke_voc/")

#create VOC folders
os.makedirs(os.path.join(voc_dir_path, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(voc_dir_path, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(voc_dir_path, "ImageSets", "Main"), exist_ok=True)

def yolo_to_voc(yolo_path, image_path, voc_path):
    try:
        with open(yolo_path, 'r') as file:
            lines = file.readlines()
    except Exception:
        lines = []

    image = Image.open(image_path)
    width, height = image.size
    image_filename = os.path.basename(image_path)

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC"
    ET.SubElement(root, "filename").text = image_filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    valid_objects = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, cx, cy, w, h = map(float, parts)
        class_id = int(class_id)

        if class_id == 0:  #skip nothjing
            continue

        valid_objects += 1

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[class_id]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = int((cx - w / 2) * width)
        ymin = int((cy - h / 2) * height)
        xmax = int((cx + w / 2) * width)
        ymax = int((cy + h / 2) * height)

        ET.SubElement(bndbox, "xmin").text = str(max(1, xmin))
        ET.SubElement(bndbox, "ymin").text = str(max(1, ymin))
        ET.SubElement(bndbox, "xmax").text = str(min(width, xmax))
        ET.SubElement(bndbox, "ymax").text = str(min(height, ymax))

    if valid_objects == 0 and REMOVE_EMPTY_ANNOTATIONS:
        return False  #remove image/annotation

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(voc_path, encoding='utf-8', xml_declaration=True)

    return True

#labels.txt
labels_txt_path = os.path.join(voc_dir_path, "labels.txt")
with open(labels_txt_path, "w") as f:
    for label in custom_labels:
        f.write(label + "\n")

#dataset splitting
subsets = ['train', 'val', 'test']
all_image_ids = []

for subset in subsets:
    label_dir = os.path.join(yolo_dir_path, subset, "labels")
    image_dir = os.path.join(yolo_dir_path, subset, "images")
    image_set_path = os.path.join(voc_dir_path, "ImageSets", "Main", f"{subset}.txt")
    image_ids = []

    if os.path.exists(label_dir):
        for file in tqdm(os.listdir(label_dir), desc=f"Processing {subset}"):
            if not file.endswith(".txt"):
                continue

            image_id = os.path.splitext(file)[0]
            yolo_path = os.path.join(label_dir, file)
            image_path = os.path.join(image_dir, image_id + ".jpg")

            if not os.path.exists(image_path):
                print(f"Missing image with id of {image_id}")
                continue

            voc_path = os.path.join(voc_dir_path, "Annotations", image_id + ".xml")

            keep = yolo_to_voc(yolo_path, image_path, voc_path)

            if not keep and REMOVE_EMPTY_ANNOTATIONS:
                continue  #skip

            #copy image to new location
            shutil.copy(image_path, os.path.join(voc_dir_path, "JPEGImages", image_id + ".jpg"))
            image_ids.append(image_id)

    #write splitting files
    with open(image_set_path, "w") as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")

    all_image_ids.extend(image_ids)

#trainval.txt
trainval_ids = []
for subset in ['train', 'val']:
    image_set_path = os.path.join(voc_dir_path, "ImageSets", "Main", f"{subset}.txt")
    if os.path.exists(image_set_path):
        with open(image_set_path) as f:
            trainval_ids.extend(line.strip() for line in f if line.strip())

trainval_path = os.path.join(voc_dir_path, "ImageSets", "Main", "trainval.txt")
with open(trainval_path, "w") as f:
    for image_id in trainval_ids:
        f.write(f"{image_id}\n")

print("Converted")
