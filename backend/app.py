# backend/app.py
import os
import json
from pathlib import Path
from io import BytesIO
from contextlib import asynccontextmanager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests as http_requests

from backend.inference import run_prediction, preprocess_image

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_TF_PATH = ASSETS_DIR / "model_tf" / "model_MobileNetV2.keras"
CLASSES_PATH = ASSETS_DIR / "classes.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend" / "dist"

# Load class names (lightweight, no TF dependency)
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """Load TF model at startup, cleanup at shutdown.
    Excluded from unit test coverage — only exercised by E2E tests
    which load the real model. Unit tests inject FakeModel directly.
    """
    import tensorflow as tf
    print("Loading TensorFlow model...")
    app.state.model = tf.keras.models.load_model(str(MODEL_TF_PATH))
    print("Model loaded successfully.")
    yield
    del app.state.model


# FastAPI app
app = FastAPI(title="Food Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve TF.js model and asset files
app.mount("/artifacts", StaticFiles(directory=str(ASSETS_DIR)), name="artifacts")


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    """Server-side inference from uploaded image."""
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    result = run_prediction(app.state.model, img, class_names)
    return JSONResponse(content=result)


@app.post("/api/predict_url")
async def predict_url(payload: dict):
    """Server-side inference from image URL."""
    url = payload.get("url", "")
    response = http_requests.get(url)
    img = Image.open(BytesIO(response.content))
    result = run_prediction(app.state.model, img, class_names)
    return JSONResponse(content=result)


@app.get("/api/classes")
async def get_classes():
    """Return the list of class names."""
    return {"classes": class_names}


# Serve frontend static files (production)
if FRONTEND_DIR.exists():  # pragma: no cover
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend SPA — returns index.html for unknown paths."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    async def root():  # pragma: no cover
        """Fallback when no frontend build exists."""
        return {"message": "Food Classifier API", "classes": class_names}
