import os
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

xml_folder = BASE_DIR / "dataset" / "NEU-DET" / "validation" / "annotations"
output_folder = BASE_DIR / "dataset" / "NEU-DET" / "validation" / "labels"

os.makedirs(output_folder, exist_ok=True)

classes = set()

for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        obj = root.find("object")
        classes.add(obj.find("name").text)

print(classes)

class_map = {c: i for i, c in enumerate(sorted(classes))}

print(class_map)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    obj = root.find("object")

    class_name = obj.find("name").text
    class_id = class_map[class_name]

    bbox = obj.find("bndbox")

    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    x_center = ((xmin + xmax) / 2) / width 
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    # Create class folder inside labels
    label_folder = os.path.join(output_folder, class_name)
    os.makedirs(label_folder, exist_ok=True)

    txt_file = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(label_folder, txt_file)

    with open(txt_path, "w") as file:
        file.write(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

print("Conversion completed")