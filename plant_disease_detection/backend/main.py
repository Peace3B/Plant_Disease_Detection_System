from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json
import time

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "models/plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]
IMAGE_SIZE = (224, 224)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess_image(img_bytes)

    preds = model.predict(img)[0]
    top_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = float(np.max(preds))

    return {
        "class": top_class,
        "confidence": confidence,
        "probabilities": {c: float(preds[i]) for i, c in enumerate(CLASS_NAMES)}
    }


@app.post("/retrain")
async def retrain():
    # Dummy mock response — can trigger an async job
    return {"message": "Retraining started", "timestamp": time.time()}


@app.get("/status")
async def status():
    return {
        "model_version": metadata.get("version", "1.0"),
        "classes": CLASS_NAMES,
        "last_retrain": metadata.get("last_retrain", "N/A")
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)