from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
from src.create_tables import SessionLocal, Inspection
from sqlalchemy.exc import SQLAlchemyError
import shutil
import os
import cv2
import uuid
import numpy as np

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

def process_image(file_path, prediction_filename):
    results = model.predict(file_path)
    result = results[0]
 
    annotated_image = result.plot()

    prediction_path = BASE_DIR / "uploads" / prediction_filename
    cv2.imwrite(str(prediction_path), annotated_image)
    
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

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty"
        )

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )


    file.file.seek(0);

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = BASE_DIR / "uploads" / unique_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction_filename = f"predicted_{uuid.uuid4()}.jpg"

    try:
        predictions = process_image(file_path, prediction_filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again."
        )
    # Save inspection in PostgreSQL
    db = SessionLocal()

    try:
        inspection = Inspection(
            image_name=file.filename,
            predictions=predictions
        )

        db.add(inspection)
        raise SQLAlchemyError("Test database failure")
        db.commit()
        db.refresh(inspection)

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to save inspection."
        )

    finally:
        db.close()

    return {
        "predicted_image": f"/uploads/{prediction_filename}",
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