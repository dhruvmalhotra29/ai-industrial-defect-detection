from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
from src.create_tables import SessionLocal, Inspection
import shutil
import os
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"], # all http methods
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=BASE_DIR/"uploads"), name="uploads")

MODEL_PATH = BASE_DIR / "models" / "best.onnx"

# Load the model once when the server starts
model = YOLO(str(MODEL_PATH))

os.makedirs(BASE_DIR / "uploads", exist_ok=True)

@app.get("/")
def home():
    return {"message": "Industrial Defect Detection API is running!"}

def process_image(file_path):
    results = model.predict(file_path)
    result = results[0]
 
    annotated_image = result.plot()
    cv2.imwrite(str(BASE_DIR / "uploads" / "predicted.jpg"), annotated_image)
    
    predictions = []

    for box in result.boxes:
        predictions.append({
            "class": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": box.xyxy.tolist()[0]
        })

    return predictions

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = BASE_DIR / f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictions = process_image(file_path)

    # Save inspection in PostgreSQL
    db = SessionLocal()

    try:
        inspection = Inspection(
            image_name=file.filename,
            predictions=predictions
        )

        db.add(inspection)
        db.commit()
        db.refresh(inspection)

    finally:
        db.close()


    return {
        "predicted_image": "/uploads/predicted.jpg",
        "predictions": predictions, 
        "inspection_id": inspection.id
    }

@app.get("/inspections")
def get_inspections():
    db = SessionLocal()

    try:
        inspections = db.query(Inspection).order_by(
            Inspection.created_at.desc()
        ).all()

        return inspections

    finally:
        db.close()