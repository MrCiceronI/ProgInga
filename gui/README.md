# Cat vs Dog — web GUI

A small FastAPI app that loads the **exported** Keras model from
`model/exports/` and lets you classify any photo from a web browser.

```
gui/
├── app.py              FastAPI routes + startup/shutdown
├── inference.py        model loading and preprocessing (the model card contract)
├── requirements.txt
├── templates/
│   ├── index.html            the upload page
│   └── partials/
│       ├── result.html       prediction card
│       └── error.html        friendly error card
├── static/style.css
└── tests/
    ├── test_app.py           real end-to-end tests
    └── render_preview.py     dumps the rendered HTML for inspection
```

## Quick start

From the repository root:

```bash
pip install -r gui/requirements.txt
python -m gui.app
```

Then open **http://127.0.0.1:8000** and pick an image. The model is loaded once
at startup (a few seconds) and reused for every request.

The Notebook's export cells must have been run at least once, so that this file
exists:

```
model/exports/model_dogs_vs_cats.keras
```

If it is missing, the page still loads but shows a clear "model could not be
loaded" panel instead of crashing.

## Configuration

All settings are optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGU_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to reach it from other devices) |
| `AGU_PORT` | `8000` | Port |
| `AGU_EXPORTS_DIR` | `model/exports` | Where to find the exported model |
| `AGU_MODEL_FORMAT` | `keras` | `keras` (`.keras` file) or `saved_model` (TF SavedModel dir) |
| `AGU_RELOAD` | off | `1` enables auto-reload while editing |

Example:

```bash
AGU_PORT=9000 AGU_MODEL_FORMAT=saved_model python -m gui.app
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | The upload page |
| `POST` | `/predict` | Form upload → HTML result card (used by the page) |
| `POST` | `/samples/{name}/predict` | Classify a bundled sample photo |
| `GET` | `/samples/{name}` | The raw sample image |
| `POST` | `/api/predict` | Form upload → JSON |
| `GET` | `/api/model` | Model metadata and preprocessing contract |
| `GET` | `/health` | Readiness probe (`503` if the model failed to load) |
| `GET` | `/docs` | Interactive OpenAPI docs |

Call the API directly:

```bash
curl -F "file=@model/my_images/cat_yasha.jpg" http://127.0.0.1:8000/api/predict
```

```json
{"filename":"cat_yasha.jpg","label":"cat","confidence":0.896,
 "dog_probability":0.104,"cat_probability":0.896,"threshold":0.5,
 "original_size":{"width":1090,"height":719},"model":"model_dogs_vs_cats.keras", ...}
```

## Preprocessing

`gui/inference.py` reproduces the export's model card exactly, so browser
predictions match the notebook:

1. decode the upload with Pillow (JPEG/PNG/BMP/GIF/WebP/TIFF);
2. apply EXIF orientation, so phone photos are not sideways;
3. convert to RGB (drops alpha, expands greyscale);
4. resize to **128×128** bilinear;
5. scale to `[0, 1]` by dividing by 255.0, as **float32**;
6. add a batch dimension → `(1, 128, 128, 3)`.

The single sigmoid output is the probability of **dog**; `>= 0.5` means dog.
The reported confidence is the probability of the winning class.

## Tests

```bash
python -m pytest gui/tests -q     # or, without pytest:
python -m gui.tests.test_app
```

The tests use the real exported model and the real photos in `model/my_images`,
so they prove the browser flow end to end. They check the preprocessing shape
and scaling, RGBA/greyscale uploads, rejection of non-images, JSON output,
determinism, path-traversal protection, and the page itself.

To eyeball the HTML without a browser:

```bash
python -m gui.tests.render_preview   # writes gui/_render/*.html
```

## Notes

- Uploads are capped at 15 MB and only whitelisted image formats are decoded.
- Sample filenames are validated so `/samples/...` cannot escape its directory.
- Inference is serialised behind a lock; Keras `predict` is not guaranteed to be
  thread-safe, and a 128×128 CNN takes milliseconds anyway.
- The model is a small CNN trained on 20 epochs of a Kaggle subset, with ~86%
  validation accuracy — expect it to be wrong on unusual photos. A photo of a
  cat that looks like a dog's snout may come back "dog"; that is the model, not
  the GUI.
