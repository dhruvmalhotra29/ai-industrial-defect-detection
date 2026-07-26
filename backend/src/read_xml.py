import os
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
train_path = BASE_DIR / "dataset" / "NEU-DET" / "train" / "annotations"

# Find the first xml file
xml_file = None

for file in os.listdir(train_path):
    if file.endswith(".xml"):
        xml_file = file
        break

print("XML file", xml_file)

tree = ET.parse(os.path.join(train_path, xml_file))
root = tree.getroot()

print("Root tag: ", root.tag)

obj = root.find("object")
bbox = obj.find("bndbox")

for point in bbox:
    print(point.tag, ":", point.text)