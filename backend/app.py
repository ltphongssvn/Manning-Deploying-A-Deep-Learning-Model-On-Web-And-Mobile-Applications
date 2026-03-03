import os
import json
import numpy as np
from pathlib import Path
from io import BytesIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests as http_requests

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_TF_PATH = ASSETS_DIR / "model_tf" / "model_MobileNetV2.keras"
CLASSES_PATH = ASSETS_DIR / "classes.json"
IMG_SIZE = (224, 224)

# Load model and classes
print("Loading TensorFlow model...")
model = tf.keras.models.load_model(str(MODEL_TF_PATH))
print("Model loaded successfully.")

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

# FastAPI app
app = FastAPI(title="Food Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve TF.js model files and frontend static files
app.mount("/artifacts", StaticFiles(directory=str(ASSETS_DIR)), name="artifacts")
if (BASE_DIR.parent / "frontend" / "build").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR.parent / "frontend" / "build" / "static")), name="static")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for MobileNetV2 inference."""
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict(img: Image.Image) -> dict:
    """Run inference and return predictions."""
    import time
    processed = preprocess_image(img)
    start = time.time()
    predictions = model.predict(processed, verbose=0)
    inference_time = time.time() - start
    probs = np.squeeze(predictions)
    results = sorted(
        zip(class_names, [float(p) for p in probs]),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "predictions": [{"class": c, "probability": p} for c, p in results],
        "inference_time_ms": round(inference_time * 1000, 2),
    }


@app.get("/")
async def root():
    return {"message": "Food Classifier API", "classes": class_names}


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    """Server-side inference from uploaded image."""
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    return JSONResponse(content=predict(img))


@app.post("/api/predict_url")
async def predict_url(payload: dict):
    """Server-side inference from image URL."""
    url = payload.get("url", "")
    response = http_requests.get(url)
    img = Image.open(BytesIO(response.content))
    return JSONResponse(content=predict(img))


@app.get("/api/classes")
async def get_classes():
    return {"classes": class_names}
