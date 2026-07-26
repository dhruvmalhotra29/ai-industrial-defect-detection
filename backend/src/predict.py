from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # backend

MODEL_PATH = BASE_DIR / "runs" / "detect" / "neu_defect_detection" / "weights" / "best.pt"

# Load trained model
model = YOLO(str(MODEL_PATH))

# Predicts an image
results = model.predict(
    source = BASE_DIR.parent / "dataset"/ "NEU-DET" / "validation" / "images" / "scratches" / "scratches_241.jpg",
    save=True,
    conf=0.25
)