from pathlib import Path
from ultralytics import YOLO
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent # /backend
DATA_YAML = BASE_DIR / "src" / "data.yaml"

# Load a pre trained YOLO model
model = YOLO("yolov8n.pt")

#Train the model
model.train(
    data=str(DATA_YAML),
    epochs=1,
    imgsz=640,
    batch=8,
    project=str(BASE_DIR / "runs"),
    name="neu_defect_detection"
)

best_model = BASE_DIR / "runs" / "detect" / "neu_defect_detection" / "weights" / "best.pt"
models_dir = BASE_DIR / "models"

models_dir.mkdir(exist_ok=True)

shutil.copy2(best_model, models_dir / "best.pt")

print("Model copied to:", models_dir / "best.pt")