"""Render the templates to disk so the HTML can be inspected without a browser.

Writes gui/_render/*.html, which is git-ignored. Run from the repo root:

    python -m gui.tests.render_preview
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gui.app import SAMPLES_DIR, app
from gui.inference import DEFAULT_EXPORTS_DIR, store

OUT_DIR = Path(__file__).resolve().parent.parent / "_render"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    captured: dict = {}
    with TestClient(app) as client:
        home = client.get("/")
        (OUT_DIR / "home.html").write_text(home.text, encoding="utf-8")
        print(f"home.html      {home.status_code}  {len(home.text):6d} bytes")
        captured["model"] = store.loaded

        if samples:
            sample = samples[0]
            up = client.post(
                "/predict", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")}
            )
            (OUT_DIR / "result.html").write_text(up.text, encoding="utf-8")
            print(f"result.html    {up.status_code}  {len(up.text):6d} bytes  ({sample.name})")

        bad = client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")})
        (OUT_DIR / "error.html").write_text(bad.text, encoding="utf-8")
        print(f"error.html     {bad.status_code}  {len(bad.text):6d} bytes")

    # The TestClient lifespan shuts down (and clears the store) on exit, so read
    # the metadata captured during the run rather than the live store.
    info = captured.setdefault("model", None)
    assert info is not None, "model was never loaded"
    print(f"\nmodel: {info.name}  input {info.width}x{info.height}  threshold {info.threshold}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
