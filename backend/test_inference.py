# backend/test_inference.py
"""Unit tests for backend.inference module.

Tests follow Given/When/Then structure and ZOMBIES mnemonic:
- Zero: minimal/empty inputs
- One: standard single input
- Boundaries: RGBA, grayscale, odd sizes, pixel value ranges
- Interfaces: return type and shape contracts
- Exceptions: invalid inputs
- Simple: basic happy path
"""
import numpy as np
import pytest
from PIL import Image

from backend.inference import preprocess_image, run_prediction


# ---------------------------------------------------------------------------
# preprocess_image: Zero / Simple
# ---------------------------------------------------------------------------
class TestPreprocessImageShape:
    """Output shape contract: always (1, 224, 224, 3) float32."""

    def test_standard_rgb_image_returns_correct_shape(self):
        # Given: a standard 512x512 RGB image
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: shape is (1, 224, 224, 3)
        assert result.shape == (1, 224, 224, 3)

    def test_output_dtype_is_float32(self):
        # Given: any RGB image
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: dtype is float32
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# preprocess_image: Boundaries
# ---------------------------------------------------------------------------
class TestPreprocessImageBoundaries:
    """Edge cases for image mode, size, and pixel values."""

    def test_rgba_image_converted_to_rgb(self):
        # Given: an RGBA image (4 channels)
        img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: output is still (1, 224, 224, 3) — alpha channel dropped
        assert result.shape == (1, 224, 224, 3)

    def test_grayscale_image_converted_to_rgb(self):
        # Given: a grayscale image (1 channel)
        img = Image.new("L", (200, 200), color=127)
        # When: preprocessed
        result = preprocess_image(img)
        # Then: output is (1, 224, 224, 3) — expanded to 3 channels
        assert result.shape == (1, 224, 224, 3)

    def test_tiny_1x1_image(self):
        # Given: the smallest possible image
        img = Image.new("RGB", (1, 1), color=(255, 255, 255))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: resized and padded to (1, 224, 224, 3)
        assert result.shape == (1, 224, 224, 3)

    def test_non_square_image(self):
        # Given: a wide rectangular image
        img = Image.new("RGB", (1000, 100), color=(64, 128, 192))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: resized to square (1, 224, 224, 3)
        assert result.shape == (1, 224, 224, 3)

    def test_pixel_values_in_mobilenet_range(self):
        # Given: an image with known pixel values (all zeros)
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: MobileNetV2 range is [-1, 1]; all-black -> all -1.0
        assert np.allclose(result, -1.0, atol=1e-6)

    def test_white_image_preprocessed_to_ones(self):
        # Given: an all-white image (255, 255, 255)
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: all-white -> 255/127.5 - 1.0 = 1.0
        assert np.allclose(result, 1.0, atol=0.01)

    def test_midgray_image_preprocessed_near_zero(self):
        # Given: a mid-gray image (127, 127, 127)
        img = Image.new("RGB", (224, 224), color=(127, 127, 127))
        # When: preprocessed
        result = preprocess_image(img)
        # Then: 127/127.5 - 1.0 ≈ -0.004 (near zero)
        assert np.all(result > -0.01) and np.all(result < 0.01)


# ---------------------------------------------------------------------------
# run_prediction: with Fake model (Test Double)
# ---------------------------------------------------------------------------
class FakeModel:
    """Test double: a fake model that returns fixed probabilities.

    Satisfies the same interface as tf.keras.Model.predict().
    """

    def __init__(self, fake_probs: list[float]):
        self._probs = np.array([fake_probs], dtype=np.float32)

    def predict(self, x, verbose=0):
        return self._probs


class TestRunPrediction:
    """Tests for run_prediction using injected FakeModel."""

    def test_returns_predictions_key(self):
        # Given: a fake model and dummy image
        model = FakeModel([0.7, 0.2, 0.1])
        img = Image.new("RGB", (224, 224))
        classes = ["apple_pie", "caesar_salad", "falafel"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: result has 'predictions' key
        assert "predictions" in result

    def test_returns_inference_time_key(self):
        # Given: a fake model
        model = FakeModel([0.5, 0.3, 0.2])
        img = Image.new("RGB", (224, 224))
        classes = ["a", "b", "c"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: result has 'inference_time_ms' key
        assert "inference_time_ms" in result
        assert isinstance(result["inference_time_ms"], float)

    def test_predictions_sorted_descending(self):
        # Given: a model that returns [0.1, 0.7, 0.2]
        model = FakeModel([0.1, 0.7, 0.2])
        img = Image.new("RGB", (224, 224))
        classes = ["apple_pie", "caesar_salad", "falafel"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: predictions sorted by probability descending
        probs = [p["probability"] for p in result["predictions"]]
        assert probs == sorted(probs, reverse=True)

    def test_predictions_contain_all_classes(self):
        # Given: 3 classes
        model = FakeModel([0.3, 0.5, 0.2])
        img = Image.new("RGB", (224, 224))
        classes = ["apple_pie", "caesar_salad", "falafel"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: all 3 classes present
        result_classes = {p["class"] for p in result["predictions"]}
        assert result_classes == set(classes)

    def test_prediction_structure(self):
        # Given: a fake model
        model = FakeModel([0.6, 0.3, 0.1])
        img = Image.new("RGB", (224, 224))
        classes = ["a", "b", "c"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: each prediction has 'class' and 'probability'
        for pred in result["predictions"]:
            assert "class" in pred
            assert "probability" in pred
            assert isinstance(pred["class"], str)
            assert isinstance(pred["probability"], float)

    def test_top_prediction_is_highest_probability(self):
        # Given: model returns highest prob for index 1 (caesar_salad)
        model = FakeModel([0.1, 0.8, 0.1])
        img = Image.new("RGB", (224, 224))
        classes = ["apple_pie", "caesar_salad", "falafel"]
        # When: prediction is run
        result = run_prediction(model, img, classes)
        # Then: top prediction is caesar_salad
        assert result["predictions"][0]["class"] == "caesar_salad"
        assert result["predictions"][0]["probability"] == pytest.approx(0.8, abs=1e-6)
