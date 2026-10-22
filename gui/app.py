"""FastAPI application: upload a photo in the browser, get cat/dog back.

Run it with::

    python -m gui.app            # or: uvicorn gui.app:app --reload

The model is loaded once at startup from ``model/exports`` and shared by every
request. Two endpoints serve the UI (``/`` and ``/predict``); the JSON API is
under ``/api`` so other tools can use the classifier too (``/docs`` has the
auto-generated schema).
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .inference import (
    DEFAULT_EXPORTS_DIR,
    DEFAULT_SAMPLES_DIRNAME,
    InvalidImageError,
    ModelNotLoadedError,
    list_samples,
    store,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = DEFAULT_EXPORTS_DIR.parent  # the model/ directory
SAMPLES_DIR = MODEL_ROOT / DEFAULT_SAMPLES_DIRNAME
TEMPLATES_DIR = BASE_DIR / "templates"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger("aguproginga.gui")


def _exports_dir() -> Path:
    """Exports directory, overridable with AGU_EXPORTS_DIR for deployments."""
    return Path(os.environ.get("AGU_EXPORTS_DIR", DEFAULT_EXPORTS_DIR)).resolve()


def _model_format() -> str:
    return os.environ.get("AGU_MODEL_FORMAT", "keras").lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    exports_dir = _exports_dir()
    try:
        loaded = store.load(exports_dir, _model_format())
        logger.info("Loaded model %s (input %sx%s)", loaded.path, loaded.width, loaded.height)
    except Exception as exc:  # noqa: BLE001 - the app must stay up to report the problem
        # Serving a clear error page beats crashing without an explanation.
        message = str(exc) or exc.__class__.__name__
        store.set_error(message)
        logger.error("Could not load model from %s: %s", exports_dir, message)
    yield
    store.clear()


app = FastAPI(
    title="Cat vs Dog classifier",
    description="Upload an image and classify it with the model exported from training_model.ipynb.",
    version=__version__,
    lifespan=lifespan,
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _context(request: Request) -> dict:
    """Template context shared by the page and the error panels."""
    loaded = store.loaded
    card = loaded.card if loaded else {}
    return {
        "request": request,
        "model_ready": loaded is not None,
        "model_error": store.error,
        "model_name": loaded.name if loaded else None,
        "input_size": f"{loaded.width}x{loaded.height}" if loaded else None,
        "threshold": loaded.threshold if loaded else 0.5,
        "metrics": card.get("validation_metrics") or {},
        "tensorflow_version": card.get("tensorflow_version"),
        "samples": list_samples(SAMPLES_DIR),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _context(request))


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness/readiness probe used by the smoke test and monitors."""
    loaded = store.loaded
    body = {"status": "ok" if loaded else "model_unavailable", "model": loaded.name if loaded else None}
    if loaded is None:
        body["error"] = store.error
        return JSONResponse(body, status_code=503)
    return JSONResponse(body)


@app.get("/samples/{name}")
async def sample_image(name: str) -> Response:
    """Serve a bundled sample photo as a raw image."""
    candidate = _sample_path(name)
    return Response(content=candidate.read_bytes(), media_type=_media_type(candidate.suffix))


def _sample_path(name: str) -> Path:
    """Resolve a sample name safely; never join user input into a path unchecked."""
    if Path(name).name != name:  # reject "..", subpaths and separators
        raise HTTPException(status_code=404, detail="Sample image not found.")
    candidate = (SAMPLES_DIR / name).resolve()
    if candidate.parent != SAMPLES_DIR.resolve() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return candidate


@app.post("/samples/{name}/predict", response_class=HTMLResponse)
async def predict_sample(request: Request, name: str) -> HTMLResponse:
    """Classify a bundled sample photo, for the one-click sample buttons."""
    candidate = _sample_path(name)
    try:
        prediction = store.predict(candidate.read_bytes())
    except (InvalidImageError, ModelNotLoadedError) as exc:
        status = 400 if isinstance(exc, InvalidImageError) else 503
        response = templates.TemplateResponse(
            request, "partials/error.html", {"request": request, "message": str(exc)}
        )
        response.status_code = status
        return response
    context = {
        "request": request,
        "result": prediction.as_dict(),
        "filename": candidate.name,
        "image_data_url": f"data:{_media_type(candidate.suffix)};base64,"
        + base64.b64encode(candidate.read_bytes()).decode("ascii"),
    }
    return templates.TemplateResponse(request, "partials/result.html", context)


def _media_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
    }.get(suffix.lower(), "application/octet-stream")


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No image was uploaded.")
    return data


@app.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Form endpoint used by the browser; returns an HTML fragment.

    Both the JS-enhanced page and a plain form post receive a self-contained
    result card, so the app degrades gracefully without JavaScript.
    """
    try:
        data = await _read_upload(file)
        prediction = store.predict(data)
    except (InvalidImageError, ModelNotLoadedError) as exc:
        status = 400 if isinstance(exc, InvalidImageError) else 503
        response = templates.TemplateResponse(
            request, "partials/error.html", {"request": request, "message": str(exc)}
        )
        response.status_code = status
        return response

    context = {
        "request": request,
        "result": prediction.as_dict(),
        "filename": file.filename or "image",
        "image_data_url": f"data:{file.content_type or 'image/jpeg'};base64,"
        + base64.b64encode(data).decode("ascii"),
    }
    return templates.TemplateResponse(request, "partials/result.html", context)


@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)) -> JSONResponse:
    """JSON API counterpart of /predict, handy for scripts and tests."""
    try:
        data = await _read_upload(file)
        prediction = store.predict(data)
    except (InvalidImageError, ModelNotLoadedError) as exc:
        status = 400 if isinstance(exc, InvalidImageError) else 503
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    body = {"filename": file.filename, **prediction.as_dict()}
    return JSONResponse(body)


@app.get("/api/model")
async def model_info() -> JSONResponse:
    """Describe the loaded model and the preprocessing contract it expects."""
    loaded = store.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail=store.error or "Model not loaded.")
    return JSONResponse(
        {
            "name": loaded.name,
            "path": str(loaded.path),
            "input": {"height": loaded.height, "width": loaded.width, "channels": 3},
            "threshold": loaded.threshold,
            "class_names": loaded.class_names,
            "card": loaded.card,
        }
    )


def main() -> None:
    """Entry point for ``python -m gui.app``."""
    import uvicorn

    uvicorn.run(
        "gui.app:app",
        host=os.environ.get("AGU_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGU_PORT", "8000")),
        reload=os.environ.get("AGU_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
