# backend/test_api.py
"""Integration tests for FastAPI endpoints.

Uses httpx AsyncClient with ASGITransport to test endpoints in-memory.
The real TF model is replaced with a FakeModel test double injected
via app.state.model, keeping tests fast and independent.

Tests follow Given/When/Then structure and ZOMBIES mnemonic:
- Zero: missing/empty inputs
- One: valid single request
- Boundaries: invalid file types, unreachable URLs
- Interfaces: response status codes and JSON structure
- Exceptions: error handling paths
- Simple: happy path
"""
import io
import numpy as np
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from backend.app import app


class FakeModel:
    """Test double: returns fixed probabilities without TensorFlow."""

    def predict(self, x, verbose=0):
        return np.array([[0.1, 0.7, 0.2]], dtype=np.float32)


@pytest.fixture(autouse=True)
def inject_fake_model():
    """Inject FakeModel into app.state before every test."""
    app.state.model = FakeModel()
    yield
    # Cleanup: remove model from state
    if hasattr(app.state, "model"):
        del app.state.model


def make_test_image_bytes(fmt: str = "JPEG") -> bytes:
    """Helper: create a minimal valid image as bytes."""
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# GET /api/classes: Simple, Interface
# ---------------------------------------------------------------------------
class TestGetClasses:
    """Tests for the /api/classes endpoint."""

    @pytest.mark.asyncio
    async def test_returns_200(self):
        # Given: the API is running
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: GET /api/classes
            response = await client.get("/api/classes")
        # Then: 200 OK
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_classes_list(self):
        # Given: the API with known classes
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: GET /api/classes
            response = await client.get("/api/classes")
        # Then: response contains expected classes
        data = response.json()
        assert "classes" in data
        assert data["classes"] == ["apple_pie", "caesar_salad", "falafel"]

    @pytest.mark.asyncio
    async def test_response_is_json(self):
        # Given/When: GET /api/classes
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/classes")
        # Then: content type is JSON
        assert "application/json" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /api/predict: Simple, Zero, Boundaries
# ---------------------------------------------------------------------------
class TestPredictUpload:
    """Tests for the /api/predict file upload endpoint."""

    @pytest.mark.asyncio
    async def test_valid_image_returns_200(self):
        # Given: a valid JPEG image
        image_bytes = make_test_image_bytes("JPEG")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST with image file
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: 200 OK
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_image_returns_predictions(self):
        # Given: a valid image
        image_bytes = make_test_image_bytes("JPEG")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST with image
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: response has predictions and inference_time_ms
        data = response.json()
        assert "predictions" in data
        assert "inference_time_ms" in data
        assert len(data["predictions"]) == 3

    @pytest.mark.asyncio
    async def test_predictions_sorted_descending(self):
        # Given: a valid image
        image_bytes = make_test_image_bytes("JPEG")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST with image
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: predictions sorted by probability descending
        data = response.json()
        probs = [p["probability"] for p in data["predictions"]]
        assert probs == sorted(probs, reverse=True)

    @pytest.mark.asyncio
    async def test_top_prediction_matches_fake_model(self):
        # Given: FakeModel returns [0.1, 0.7, 0.2] -> caesar_salad highest
        image_bytes = make_test_image_bytes("JPEG")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST with image
            response = await client.post(
                "/api/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        # Then: top prediction is caesar_salad at 0.7
        data = response.json()
        assert data["predictions"][0]["class"] == "caesar_salad"

    @pytest.mark.asyncio
    async def test_png_image_accepted(self):
        # Given: a PNG image (different format)
        image_bytes = make_test_image_bytes("PNG")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST with PNG
            response = await client.post(
                "/api/predict",
                files={"file": ("test.png", image_bytes, "image/png")},
            )
        # Then: still works, 200 OK
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_file_returns_422(self):
        # Given: no file attached
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # When: POST without file
            response = await client.post("/api/predict")
        # Then: 422 Unprocessable Entity (FastAPI validation error)
        assert response.status_code == 422
