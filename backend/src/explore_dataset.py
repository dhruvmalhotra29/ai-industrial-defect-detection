import os
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
train_path = BASE_DIR / "dataset" / "NEU-DET" / "train"
val_path = BASE_DIR / "dataset" / "NEU-DET" / "validation"

print("Train folders:")
print(os.listdir(train_path))

print("\n Validation folders:")
print(os.listdir(train_path))

#Count images
def count_images(folder):
    count = 0
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    return count

print("\nTraining images:", count_images(train_path))
print("\nValidation images:",count_images(val_path))
