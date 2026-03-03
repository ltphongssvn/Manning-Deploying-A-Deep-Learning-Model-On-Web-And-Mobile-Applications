# backend/inference.py
import time
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess a PIL image for MobileNetV2 inference.

    Converts to RGB, resizes to 224x224, applies MobileNetV2
    preprocessing (scale pixels to [-1, 1]).

    Args:
        img: PIL Image in any mode/size.

    Returns:
        numpy array of shape (1, 224, 224, 3) with float32 values in [-1, 1].
    """
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale [0, 255] -> [-1, 1]
    img_array = img_array / 127.5 - 1.0
    return np.expand_dims(img_array, axis=0)


def run_prediction(model, img: Image.Image, class_names: list[str]) -> dict:
    """Run inference on a PIL image using the given model.

    Args:
        model: A loaded TF/Keras model with a .predict() method.
        img: PIL Image to classify.
        class_names: List of class label strings.

    Returns:
        dict with 'predictions' (sorted desc by probability) and
        'inference_time_ms'.
    """
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
