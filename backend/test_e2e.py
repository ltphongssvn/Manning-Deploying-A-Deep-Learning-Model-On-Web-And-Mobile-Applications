# backend/test_e2e.py
"""End-to-end integration tests with real TensorFlow model.

These tests load the actual MobileNetV2 model and verify the full
inference pipeline: HTTP request → FastAPI → real preprocessing →
real model prediction → JSON response.

Marked with @pytest.mark.e2e — excluded from default pytest runs
(pre-commit hooks stay fast). Run explicitly with:
    uv run pytest -m e2e -v

Tests follow Given/When/Then structure and ZOMBIES mnemonic:
- Simple: real model loads and predicts
- Interfaces: API returns correct structure with real model
- Boundaries: different image sizes, formats, color modes
- Zero: verify probabilities sum to ~1.0
"""
import io
import os
import numpy as np
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

# Conditionally import TF — only when e2e tests actually run
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@pytest.fixture(scope="module")
def real_model():
    """Load the real TF model once for all E2E tests in this module."""
    import tensorflow as tf
    from backend.app import app, BASE_DIR

    model_path = BASE_DIR / "assets" / "model_tf" / "model_MobileNetV2.keras"
    assert model_path.exists(), f"Model file not found: {model_path}"
    model = tf.keras.models.load_model(str(model_path))
    app.state.model = model
    yield app
    del app.state.model


def make_test_image(
    size: tuple = (224, 224), color: tuple = (128, 128, 128), mode: str = "RGB"
) -> bytes:
    """Create a test image as JPEG bytes."""
    img = Image.new(mode, size, color=color)
    if mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# E2E: Model Loading
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestModelLoading:
    """Verify the real TF model loads and has expected properties."""

    def test_model_loads_successfully(self, real_model):
        # Given: the real model fixture
        # Then: model is loaded on app.state
        assert hasattr(real_model.state, "model")
        assert real_model.state.model is not None

    def test_model_has_correct_input_shape(self, real_model):
        # Given: the loaded model
        model = real_model.state.model
        # Then: input shape matches MobileNetV2 (None, 224, 224, 3)
        input_shape = model.input_shape
        assert input_shape == (None, 224, 224, 3)

    def test_model_has_correct_output_shape(self, real_model):
        # Given: the loaded model
        model = real_model.state.model
        # Then: output shape matches 3 classes (None, 3)
        output_shape = model.output_shape
        assert output_shape == (None, 3)


# ---------------------------------------------------------------------------
# E2E: Real Predictions via API
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestRealPredictions:
    """Test full inference pipeline with real model through API endpoints."""

    @pytest.mark.asyncio
    async def test_predict_returns_200_with_real_model(self, real_model):
        # Given: a test image and the real model
        image_bytes = make_test_image()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: POST to /api/predict
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: 200 OK
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_predictions_have_all_three_classes(self, real_model):
        # Given: a test image
        image_bytes = make_test_image()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: all 3 classes present
        data = response.json()
        classes = {p["class"] for p in data["predictions"]}
        assert classes == {"apple_pie", "caesar_salad", "falafel"}

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_approximately_one(self, real_model):
        # Given: a test image
        image_bytes = make_test_image()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: probabilities sum to ~1.0 (softmax output)
        data = response.json()
        total = sum(p["probability"] for p in data["predictions"])
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_inference_time_is_reported(self, real_model):
        # Given: a test image
        image_bytes = make_test_image()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: inference_time_ms is positive
        data = response.json()
        assert data["inference_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_predictions_sorted_descending_real_model(self, real_model):
        # Given: a test image
        image_bytes = make_test_image()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: sorted descending by probability
        data = response.json()
        probs = [p["probability"] for p in data["predictions"]]
        assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# E2E: Boundary Conditions with Real Model
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestRealModelBoundaries:
    """Test edge case inputs through the full pipeline with real model."""

    @pytest.mark.asyncio
    async def test_tiny_image_still_predicts(self, real_model):
        # Given: a very small 10x10 image
        image_bytes = make_test_image(size=(10, 10))
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("tiny.jpg", image_bytes, "image/jpeg")},
            )
        # Then: still returns valid predictions
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3

    @pytest.mark.asyncio
    async def test_large_image_still_predicts(self, real_model):
        # Given: a large 2000x2000 image
        image_bytes = make_test_image(size=(2000, 2000))
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict
            response = await client.post(
                "/api/predict",
                files={"file": ("large.jpg", image_bytes, "image/jpeg")},
            )
        # Then: still returns valid predictions
        assert response.status_code == 200
        data = response.json()
        total = sum(p["probability"] for p in data["predictions"])
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_png_format_with_real_model(self, real_model):
        # Given: a PNG image
        img = Image.new("RGB", (224, 224), color=(64, 128, 192))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.read()
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: predict with PNG
            response = await client.post(
                "/api/predict",
                files={"file": ("test.png", png_bytes, "image/png")},
            )
        # Then: valid prediction
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_classes_still_works_with_model_loaded(self, real_model):
        # Given: real model loaded
        async with AsyncClient(
            transport=ASGITransport(app=real_model), base_url="http://test"
        ) as client:
            # When: GET /api/classes
            response = await client.get("/api/classes")
        # Then: returns expected classes
        data = response.json()
        assert data["classes"] == ["apple_pie", "caesar_salad", "falafel"]
