"""
FastAPI backend for plant disease detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import time
import asyncio
from pathlib import Path
import shutil
import zipfile
import io
import json

from prediction import DiseasePredictor
from model import PlantDiseaseModel
from preprocessing import ImagePreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for plant disease classification using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "models/plant_disease_model.h5"
UPLOAD_DIR = Path("uploads/retrain")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global variables
predictor = None
model_uptime_start = None
request_count = 0
total_latency = 0
retraining_status = {"status": "idle", "progress": 0, "message": ""}


# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3_predictions: dict
    processing_time: float


class ModelStatus(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    average_latency: float
    model_path: str
    classes: List[str]


class RetrainRequest(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor, model_uptime_start
    
    try:
        predictor = DiseasePredictor(MODEL_PATH)
        model_uptime_start = time.time()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "upload_data": "/upload-training-data",
            "retrain": "/retrain",
            "status": "/status",
            "model_info": "/model-info"
        }
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


# Single prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from a single plant image
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        Prediction results
    """
    global request_count, total_latency
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        start_time = time.time()
        contents = await file.read()
        
        # Make prediction
        result = predictor.predict(contents)
        
        # Calculate latency
        processing_time = time.time() - start_time
        request_count += 1
        total_latency += processing_time
        
        result['processing_time'] = processing_time
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction
@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict disease for multiple images
    
    Args:
        files: List of image files
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    try:
        results = []
        start_time = time.time()
        
        for file in files:
            contents = await file.read()
            result = predictor.predict(contents)
            result['filename'] = file.filename
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "total_images": len(files),
            "processing_time": processing_time,
            "average_time_per_image": processing_time / len(files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Upload training data
@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload training data (ZIP file with images organized by class)
    
    Expected structure:
    - images.zip
      ├── class1/
      │   ├── image1.jpg
      │   └── image2.jpg
      └── class2/
          ├── image1.jpg
          └── image2.jpg
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        # Read ZIP file
        contents = await file.read()
        
        # Extract ZIP
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extract_dir = UPLOAD_DIR / timestamp
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Count uploaded files
        image_count = len(list(extract_dir.rglob('*.jpg'))) + len(list(extract_dir.rglob('*.png')))
        classes = [d.name for d in extract_dir.iterdir() if d.is_dir()]
        
        return {
            "message": "Training data uploaded successfully",
            "upload_id": timestamp,
            "total_images": image_count,
            "classes": classes,
            "path": str(extract_dir)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


# Trigger retraining
@app.post("/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    request: RetrainRequest = RetrainRequest()
):
    """
    Trigger model retraining with uploaded data
    
    This runs in the background to avoid blocking
    """
    global retraining_status
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if retraining_status["status"] == "running":
        return {
            "message": "Retraining already in progress",
            "status": retraining_status
        }
    
    # Check if training data exists
    data_dirs = list(UPLOAD_DIR.iterdir())
    if not data_dirs:
        raise HTTPException(status_code=400, detail="No training data uploaded")
    
    # Use most recent upload
    latest_dir = max(data_dirs, key=lambda p: p.stat().st_mtime)
    
    # Start retraining in background
    background_tasks.add_task(
        run_retraining,
        str(latest_dir),
        request.epochs,
        request.batch_size
    )
    
    return {
        "message": "Retraining started",
        "data_directory": str(latest_dir),
        "epochs": request.epochs,
        "status": "started"
    }


# Background retraining task
async def run_retraining(data_dir: str, epochs: int, batch_size: int):
    """Run retraining in background"""
    global predictor, retraining_status, model_uptime_start
    
    try:
        retraining_status = {
            "status": "running",
            "progress": 0,
            "message": "Initializing retraining..."
        }
        
        # Load current model
        model = PlantDiseaseModel(
            num_classes=predictor.model.output_shape[-1],
            model_path=MODEL_PATH
        )
        
        retraining_status["message"] = "Training model..."
        retraining_status["progress"] = 25
        
        # Retrain
        history = model.retrain(
            new_data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            fine_tune=True
        )
        
        retraining_status["progress"] = 75
        retraining_status["message"] = "Saving model..."
        
        # Save retrained model
        model.save_model(MODEL_PATH)
        
        # Reload predictor
        predictor = DiseasePredictor(MODEL_PATH)
        model_uptime_start = time.time()
        
        retraining_status = {
            "status": "completed",
            "progress": 100,
            "message": "Retraining completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        retraining_status = {
            "status": "failed",
            "progress": 0,
            "message": f"Retraining failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


# Get retraining status
@app.get("/retrain-status")
async def get_retrain_status():
    """Get current retraining status"""
    return retraining_status


# Model status
@app.get("/status", response_model=ModelStatus)
async def get_status():
    """Get model status and metrics"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    uptime = time.time() - model_uptime_start
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    return {
        "status": "running",
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "average_latency": avg_latency,
        "model_path": MODEL_PATH,
        "classes": predictor.class_names
    }


# Model information
@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return info


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )