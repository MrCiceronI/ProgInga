"""Model loading and inference for the exported cat/dog classifier.

Everything here mirrors the preprocessing contract recorded in the export's
model card (``model/exports/model_dogs_vs_cats_card.json``):

* RGB, 3 channels
* resized to 128x128 with bilinear filtering
* pixel values scaled to [0, 1] by dividing by 255.0
* float32 array of shape (1, 128, 128, 3)
* signature output is the probability of the *dog* class, threshold 0.5

Keeping this in one module means the browser, the API and any test all apply
exactly the same transform the model was trained with.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Default location of the exported artifacts, relative to this file so the app
# works no matter which directory uvicorn is started from.
DEFAULT_EXPORTS_DIR = Path(__file__).resolve().parent.parent / "model" / "exports"
MODEL_FILENAME = "model_dogs_vs_cats.keras"
CARD_FILENAME = "model_dogs_vs_cats_card.json"
SAVED_MODEL_DIRNAME = "saved_model_dogs_vs_cats"

# Formats the browser may send. PIL decodes far more, but this keeps the
# attack surface predictable and matches what the notebook advertises.
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "GIF", "WEBP", "TIFF"}

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # reject absurd uploads before decoding


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before a model is available."""


class InvalidImageError(ValueError):
    """Raised when an uploaded payload is not a usable image."""


@dataclass
class Prediction:
    """One classification result, ready to be rendered or serialised."""

    label: str
    confidence: float
    dog_probability: float
    cat_probability: float
    threshold: float
    original_size: tuple[int, int]
    width: int
    height: int
    model_name: str

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "dog_probability": round(self.dog_probability, 6),
            "cat_probability": round(self.cat_probability, 6),
            "threshold": self.threshold,
            "original_size": {"width": self.original_size[0], "height": self.original_size[1]},
            # Flat keys for the template, which must not reach into the object.
            "width": self.width,
            "height": self.height,
            "model": self.model_name,
            # Percentages are pre-computed so the template stays logic-free.
            "dog_percent": round(self.dog_probability * 100, 2),
            "cat_percent": round(self.cat_probability * 100, 2),
            "confidence_percent": round(self.confidence * 100, 2),
        }


@dataclass
class LoadedModel:
    """A Keras model plus the metadata the UI wants to display."""

    model: object
    input_size: tuple[int, int] = (128, 128)
    threshold: float = 0.5
    class_names: dict[str, str] = field(default_factory=lambda: {"0": "cat", "1": "dog"})
    card: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.path.name if self.path else "in-memory model"

    @property
    def width(self) -> int:
        return self.input_size[1]

    @property
    def height(self) -> int:
        return self.input_size[0]

    def predict(self, image_bytes: bytes) -> Prediction:
        """Classify raw image bytes uploaded from a browser."""
        batch, original_size = self.preprocess(image_bytes)
        raw = self.model.predict(batch, verbose=0)
        dog_probability = float(np.asarray(raw).reshape(-1)[0])
        # Guard against a model that returns NaN/inf so the UI never shows "--".
        if not np.isfinite(dog_probability):
            raise ModelNotLoadedError("The model returned a non-finite score for this image.")
        dog_probability = min(max(dog_probability, 0.0), 1.0)
        cat_probability = 1.0 - dog_probability
        label = "dog" if dog_probability >= self.threshold else "cat"
        confidence = dog_probability if label == "dog" else cat_probability
        return Prediction(
            label=label,
            confidence=confidence,
            dog_probability=dog_probability,
            cat_probability=cat_probability,
            threshold=self.threshold,
            original_size=original_size,
            width=self.width,
            height=self.height,
            model_name=self.name,
        )

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        """Turn upload bytes into a (1, H, W, 3) float32 batch in [0, 1]."""
        if not image_bytes:
            raise InvalidImageError("The uploaded file is empty.")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise InvalidImageError(
                f"Image is too large ({len(image_bytes) / 1e6:.1f} MB); the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.format not in ALLOWED_FORMATS:
                    raise InvalidImageError(
                        f"Unsupported image format: {img.format}. "
                        f"Use one of: {', '.join(sorted(ALLOWED_FORMATS))}."
                    )
                original_size = img.size
                img = ImageOps.exif_transpose(img)  # honour phone orientation
                img = img.convert("RGB")            # drop alpha, collapse greyscale
                img = img.resize((self.width, self.height), Image.BILINEAR)
                array = np.asarray(img, dtype=np.float32) / 255.0
        except UnidentifiedImageError as exc:
            raise InvalidImageError("That file is not a readable image.") from exc
        except OSError as exc:  # truncated / corrupt payloads
            raise InvalidImageError(f"Could not decode the image: {exc}") from exc
        if array.shape != (self.height, self.width, 3):
            raise InvalidImageError(f"Unexpected image shape after preprocessing: {array.shape}")
        return np.expand_dims(array, axis=0), original_size


def read_card(exports_dir: Path) -> dict:
    """Load the model card if present; it documents the contract we follow."""
    card_path = Path(exports_dir) / CARD_FILENAME
    try:
        return json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_model_path(exports_dir: Path, fmt: str = "keras") -> Path:
    """Pick the exported artifact to load, failing with a helpful message."""
    exports_dir = Path(exports_dir)
    if fmt == "saved_model":
        candidate = exports_dir / SAVED_MODEL_DIRNAME
    else:
        candidate = exports_dir / MODEL_FILENAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"No exported model found at {candidate}. Run the export cells in "
            f"model/training_model.ipynb first, or point --exports-dir elsewhere."
        )
    return candidate


def load_model(exports_dir: Path = DEFAULT_EXPORTS_DIR, fmt: str = "keras") -> LoadedModel:
    """Load the exported model and align its settings with the model card."""
    # Imported lazily so `--help`, linters and unit tests of pure helpers do not
    # pay the multi-second TensorFlow import cost.
    import tensorflow as tf

    path = resolve_model_path(exports_dir, fmt)
    card = read_card(exports_dir)
    model = tf.keras.models.load_model(path)

    loaded = LoadedModel(model=model, card=card, path=path)
    input_spec = card.get("input") or {}
    if input_spec.get("height") and input_spec.get("width"):
        loaded.input_size = (int(input_spec["height"]), int(input_spec["width"]))
    output_spec = card.get("output") or {}
    if output_spec.get("decision_threshold") is not None:
        loaded.threshold = float(output_spec["decision_threshold"])
    if output_spec.get("class_names"):
        loaded.class_names = {str(k): str(v) for k, v in output_spec["class_names"].items()}

    # If the card is missing, fall back to the shape baked into the model itself.
    if not input_spec.get("height"):
        shape = model.input_shape
        if isinstance(shape, list):  # multi-input models
            shape = shape[0]
        if len(shape) == 4 and shape[1] and shape[2]:
            loaded.input_size = (int(shape[1]), int(shape[2]))
    return loaded


class ModelStore:
    """Holds the single loaded model shared by every request.

    Keras models are not guaranteed to be safe for concurrent `predict` calls
    from multiple threads, so predictions are serialised with a lock. Inference
    on a 128x128 CNN is milliseconds, so this costs nothing in practice.
    """

    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def load(self, exports_dir: Path = DEFAULT_EXPORTS_DIR, fmt: str = "keras") -> LoadedModel:
        loaded = load_model(exports_dir, fmt)
        with self._lock:
            self._loaded = loaded
            self._error = None
        return loaded

    def set_error(self, message: str) -> None:
        self._error = message

    def clear(self) -> None:
        """Drop the model (called on shutdown to release memory)."""
        with self._lock:
            self._loaded = None

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    @property
    def error(self) -> str | None:
        return self._error

    def predict(self, image_bytes: bytes) -> Prediction:
        if self._loaded is None:
            raise ModelNotLoadedError(self._error or "The model has not been loaded yet.")
        with self._lock:
            return self._loaded.predict(image_bytes)


# Module-level store used by the FastAPI app.
store = ModelStore()

# Reference images shipped with the repo, offered as one-click samples.
DEFAULT_SAMPLES_DIRNAME = "my_images"


def list_samples(images_dir: Path) -> list[dict]:
    """Index the sample photos bundled next to the exports, if any."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    samples = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}:
            continue
        samples.append({"name": path.name, "stem": path.stem.replace("_", " ")})
    return samples
