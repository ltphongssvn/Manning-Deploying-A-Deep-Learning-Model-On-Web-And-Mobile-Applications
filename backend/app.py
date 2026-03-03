import os
import json
import time
import numpy as np
from pathlib import Path
from io import BytesIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests as http_requests

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_TF_PATH = ASSETS_DIR / "model_tf" / "model_MobileNetV2.keras"
CLASSES_PATH = ASSETS_DIR / "classes.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend" / "dist"
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

# Serve TF.js model and asset files
app.mount("/artifacts", StaticFiles(directory=str(ASSETS_DIR)), name="artifacts")


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict(img: Image.Image) -> dict:
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


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    return JSONResponse(content=predict(img))


@app.post("/api/predict_url")
async def predict_url(payload: dict):
    url = payload.get("url", "")
    response = http_requests.get(url)
    img = Image.open(BytesIO(response.content))
    return JSONResponse(content=predict(img))


@app.get("/api/classes")
async def get_classes():
    return {"classes": class_names}


# Serve frontend static files (production)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Food Classifier API", "classes": class_names}
