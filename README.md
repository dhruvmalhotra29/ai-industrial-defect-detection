# Industrial Surface Defect Detection

An end-to-end computer vision system for automated industrial surface defect inspection using **YOLOv8, ONNX Runtime, FastAPI, React, and PostgreSQL**.

## Features

- Automated industrial surface defect detection
- YOLOv8-based object detection
- NEU-DET dataset for model training
- ONNX model export for production inference
- ONNX Runtime CPU inference
- Image upload and annotated predictions
- Defect confidence scores and bounding boxes
- Automated PDF inspection reports
- PostgreSQL-based inspection history
- React + Vite frontend
- FastAPI backend

## Tech Stack

- **Machine Learning:** Python, YOLOv8, Ultralytics, OpenCV
- **Model Serving:** ONNX, ONNX Runtime
- **Backend:** FastAPI, SQLAlchemy
- **Database:** PostgreSQL
- **Frontend:** React, Vite
- **Deployment:** Docker, Google Cloud Run

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

**Project Walkthrough Video:** Coming Soon

**Live Application:** Coming Soon

---

## Author

- [@dhruvmalhotra29](https://github.com/dhruvmalhotra29)