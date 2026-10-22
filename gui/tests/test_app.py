"""Tests for the cat/dog web GUI.

Run from the repository root::

    python -m pytest gui/tests -q          # if pytest is installed
    python -m gui.tests.test_app           # otherwise, uses plain asserts

These tests use the real exported model in ``model/exports`` and the real
sample photos in ``model/my_images``, so passing them means the browser flow
genuinely works. The model is loaded once and reused.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from gui.app import MODEL_ROOT, SAMPLES_DIR, app
from gui.inference import DEFAULT_EXPORTS_DIR, load_model, store


def _png_bytes(size=(64, 64), color=(200, 120, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_files() -> list[Path]:
    return sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.is_dir() else []


def test_artifacts_exist() -> None:
    """The model the GUI is supposed to serve is actually on disk."""
    assert (DEFAULT_EXPORTS_DIR / "model_dogs_vs_cats.keras").is_file()
    assert _sample_files(), f"no sample images in {SAMPLES_DIR}"


def test_preprocessing_contract() -> None:
    """Resize, RGB conversion and [0, 1] scaling must match the model card."""
    loaded = store.loaded or store.load(DEFAULT_EXPORTS_DIR)
    batch, original_size = loaded.preprocess(_png_bytes(size=(300, 500)))
    assert batch.shape == (1, 128, 128, 3), batch.shape
    assert batch.dtype.name == "float32", batch.dtype
    assert 0.0 <= batch.min() and batch.max() <= 1.0
    assert original_size == (300, 500)


def test_rgba_and_grayscale_are_converted() -> None:
    """Web screenshots are often RGBA; greyscale photos still must work."""
    loaded = store.loaded or store.load(DEFAULT_EXPORTS_DIR)
    for mode in ("RGBA", "L", "P"):
        buffer = io.BytesIO()
        Image.new(mode, (40, 40)).save(buffer, format="PNG")
        batch, _ = loaded.preprocess(buffer.getvalue())
        assert batch.shape == (1, 128, 128, 3), mode


def test_rejects_non_image() -> None:
    loaded = store.loaded or store.load(DEFAULT_EXPORTS_DIR)
    for payload in (b"not an image at all", b""):
        try:
            loaded.preprocess(payload)
        except Exception as exc:  # InvalidImageError
            assert "image" in str(exc).lower() or "empty" in str(exc).lower()
        else:
            raise AssertionError(f"accepted invalid payload: {payload!r}")


def test_samples_are_classified() -> None:
    """Every bundled sample must produce a usable prediction."""
    loaded = store.loaded or store.load(DEFAULT_EXPORTS_DIR)
    for path in _sample_files():
        prediction = loaded.predict(path.read_bytes())
        assert prediction.label in {"cat", "dog"}, path.name
        assert 0.0 <= prediction.dog_probability <= 1.0
        assert abs(prediction.dog_probability + prediction.cat_probability - 1.0) < 1e-6
        print(f"  {path.name:24s} -> {prediction.label:3s} ({prediction.confidence:.1%})")


def test_predictions_are_deterministic() -> None:
    loaded = store.loaded or store.load(DEFAULT_EXPORTS_DIR)
    data = _sample_files()[0].read_bytes()
    first = loaded.predict(data)
    second = loaded.predict(data)
    assert first.dog_probability == second.dog_probability


def test_http_routes() -> None:
    """Drive the endpoints the browser actually calls."""
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200, page.text
        assert "Cat vs Dog" in page.text
        assert "could not be loaded" not in page.text, "model failed to load at startup"

        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok"

        info = client.get("/api/model").json()
        assert info["input"] == {"height": 128, "width": 128, "channels": 3}
        assert info["threshold"] == 0.5

        sample = _sample_files()[0]
        upload = client.post("/predict", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")})
        assert upload.status_code == 200, upload.text
        for needle in ("Prediction", "confidence", "Dog", "Cat"):
            assert needle in upload.text, needle

        api = client.post("/api/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
        assert api.status_code == 200, api.text
        body = api.json()
        assert body["label"] in {"cat", "dog"}
        assert json.dumps(body)  # must be JSON serialisable

        bad = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert bad.status_code == 400, bad.text

        # Sample buttons/deep link
        sample_get = client.get(f"/samples/{sample.name}")
        assert sample_get.status_code == 200
        assert sample_get.headers["content-type"].startswith("image/")
        sample_pred = client.post(f"/samples/{sample.name}/predict")
        assert sample_pred.status_code == 200, sample_pred.text

        # Path traversal must not escape the samples directory.
        assert client.post("/samples/..%2F..%2Fmodel_dogs_vs_cats_card.json/predict").status_code == 404


if __name__ == "__main__":
    # Load the model once up front so the first test does not pay the cost twice.
    store.load(DEFAULT_EXPORTS_DIR)
    tests = [
        test_artifacts_exist,
        test_preprocessing_contract,
        test_rgba_and_grayscale_are_converted,
        test_rejects_non_image,
        test_samples_are_classified,
        test_predictions_are_deterministic,
        test_http_routes,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print(f"\nALL {len(tests)} GUI TESTS PASSED")
