from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

image_path = BASE_DIR / "dataset" / "NEU-DET" / "train" / "images" / "crazing" / "crazing_1.jpg"

img =  Image.open(str(image_path))

print("Width:", img.width)
print("Height:", img.height)