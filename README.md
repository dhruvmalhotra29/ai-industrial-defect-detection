# Industrial Surface Defect Detection

An end-to-end computer vision system for automated industrial surface defect inspection using **YOLOv8, ONNX Runtime, FastAPI, React, and PostgreSQL**.

## Features

- Automated industrial surface defect detection
- YOLOv8-based object detection
- NEU-DET dataset for model training
- ONNX model export for production inference
- ONNX Runtime CPU inference
- Image upload and annotated predictions
- Image upload validation (file size, MIME type, and image integrity)
- API rate limiting for prediction and inspection history endpoints
- Defect confidence scores and bounding boxes
- Automated PDF inspection reports
- PostgreSQL-based inspection history
- React + Vite frontend
- FastAPI backend

## Tech Stack

- **Machine Learning:** Python, YOLOv8, Ultralytics, OpenCV
- **Model Serving:** ONNX, ONNX Runtime
- **Backend:** FastAPI, SQLAlchemy
- **Database:** PostgreSQL, Neon
- **Frontend:** React, Vite
- **Deployment:** AWS EC2 (Backend), AWS S3 (Frontend)

## Architecture

```text
                React + Vite
                     │
                     ▼
                FastAPI API
                     │
                     ▼
              ONNX Runtime
                     │
                     ▼
                YOLOv8 Model
                     │
              ┌──────┴──────┐
              ▼             ▼
         Predictions    Annotated Image
              │
        ┌─────┴─────┐
        ▼           ▼
   PostgreSQL    PDF Report
```

## Model Pipeline

The model is trained using the **NEU Surface Defect Database (NEU-DET)**.

The trained YOLOv8 model is exported to ONNX for production-oriented inference:

```text
NEU-DET Dataset
      ↓
YOLOv8 Training
      ↓
   best.pt
      ↓
 ONNX Export
      ↓
  best.onnx
      ↓
ONNX Runtime
      ↓
Defect Detection
```

ONNX Runtime enables CPU-based model inference while separating the training environment from the production inference environment.

## Dataset

The model is trained using the [NEU Surface Defect Database (NEU-DET)](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database), available on Kaggle.

The dataset contains six categories of steel surface defects:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Run Locally (Optional)

Clone the repository and configure the required environment variables for the backend.

Ensure PostgreSQL is running and create the required database before launching the application.

Create a `.env` file inside the `backend` directory with the required configuration variables.

Create and activate a virtual environment before installing the backend dependencies.

### Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Database Setup

After creating the PostgreSQL database, run the database initialization script:

```bash
python create_tables.py
```

This creates the required PostgreSQL tables for storing inspection history.

### Start Backend

```bash
python -m uvicorn src.main:app --reload
```

### Start Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The backend runs at `http://127.0.0.1:8000` and the frontend runs at `http://localhost:5173`.

## Environment Variables

Create `backend/.env`:

```env
FRONTEND_URL=http://localhost:5173
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/industrial_defect_detection
```

Sensitive environment variables are excluded from version control.

## Demo

**Project Walkthrough Video:** [Watch Here](https://drive.google.com/drive/folders/1WbuzDOgPToEDvALv8CVddE977xD01p_f?usp=drive_link)

**Live Application:** [Try it here](http://industrial-defect-frontend-154560316129-ap-south-1-an.s3-website.ap-south-1.amazonaws.com/)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Check API health |
| POST | /predict | Upload an image and detect defects |
| GET | /inspections | Retrieve inspection history |

## Future Improvements

- Configure HTTPS for secure communication between the frontend and backend.

## Author

- [@dhruvmalhotra29](https://github.com/dhruvmalhotra29)