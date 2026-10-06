# Apache-2.0
"""A small HTTP server around a Detector: pictures in, detections out.

    easydetect serve model=best.onnx                # http://127.0.0.1:8000
    easydetect serve model=best.xml host=0.0.0.0 port=9000 task=pose

    curl -F image=@bus.jpg http://127.0.0.1:8000/predict
    curl --data-binary @bus.jpg -H "Content-Type: image/jpeg" \\
         "http://127.0.0.1:8000/predict?conf=0.4&draw=1" -o drawn.jpg

``POST /predict`` takes the picture as the raw body or as the ``image`` field
of a form, and answers with ``r.summary()`` as JSON — or, with ``draw=1``,
the drawn picture as a JPEG. ``conf``, ``iou``, ``max_det`` and ``classes``
(``0,2``) ride in the query string. ``GET /health`` says which model is
loaded and its classes.

Only the standard library: it runs wherever the model does, ONNX Runtime on
a Raspberry Pi included. It listens on 127.0.0.1 unless told ``host=0.0.0.0``
— there is no authentication, so put anything public behind a proxy that has.
"""

from __future__ import annotations

import json
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

MAX_BYTES = 32 * 1024 * 1024  # a picture, not a video


def _picture(content_type: str, body: bytes) -> bytes:
    """The image bytes of a request: the body itself, or a form's ``image`` field."""
    if content_type.startswith("multipart/form-data"):
        message = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode() + body)
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "image":
                return part.get_payload(decode=True) or b""
        raise ValueError("the form has no 'image' field")
    return body


def _options(query: dict[str, list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "conf" in query:
        out["conf"] = float(query["conf"][0])
    if "iou" in query:
        value = query["iou"][0]
        out["iou"] = None if value.lower() in ("none", "off") else float(value)
    if "max_det" in query:
        out["max_det"] = int(query["max_det"][0])
    if "classes" in query:
        out["classes"] = [int(c) for c in query["classes"][0].split(",") if c.strip()]
    return out


def make_handler(model, name: str):
    """The request handler class for one loaded Detector."""
    import cv2

    class Handler(BaseHTTPRequestHandler):
        server_version = "easydetect"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def log_message(self, fmt, *args):  # one short line a request
            print(f"[easydetect serve] {self.address_string()} {fmt % args}")

        def do_GET(self):  # noqa: N802 — the stdlib's naming
            if urlparse(self.path).path != "/health":
                return self._json(404, {"error": "try POST /predict or GET /health"})
            names = model.names or (model.predictor.names if model.predictor else {})
            self._json(200, {"status": "ok", "model": name, "task": model.task,
                             "names": {str(k): v for k, v in names.items()}})

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path != "/predict":
                return self._json(404, {"error": "POST pictures to /predict"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BYTES:
                return self._json(413 if length else 400,
                                  {"error": f"send one picture of up to {MAX_BYTES >> 20} MB"})
            query = parse_qs(url.query)
            try:
                data = _picture(self.headers.get("Content-Type", ""), self.rfile.read(length))
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("could not read that picture")
                options = _options(query)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            started = time.perf_counter()
            result = model(img, verbose=False, **options)[0]
            if query.get("draw", ["0"])[0] in ("1", "true", "yes"):
                ok, jpg = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
                return self._send(200, jpg.tobytes(), "image/jpeg")
            h, w = img.shape[:2]
            self._json(200, {"detections": result.summary(), "width": w, "height": h,
                             "ms": round((time.perf_counter() - started) * 1e3, 1),
                             "speed": {k: round(v, 1) for k, v in result.speed.items()}})

    return Handler


def serve(model: str = "dfine-s", host: str = "127.0.0.1", port: int = 8000,
          device: str = "AUTO", task: str = "detect", backend: str | None = None,
          ready=None, pose_model: str = "s", pose_flip: bool = True) -> None:
    """Load ``model`` once and answer requests until interrupted. ``ready`` is
    called with the bound ``(host, port)`` once it listens (port 0: any free)."""
    from .model import Detector

    detector = Detector(model, device=device, task=task, backend=backend, verbose=False)
    detector.pose_model, detector.pose_flip = pose_model, pose_flip
    # load (and, for a named model, download) now, not on the first request
    detector(np.zeros((32, 32, 3), np.uint8), verbose=False)
    server = ThreadingHTTPServer((host, int(port)), make_handler(detector, str(model)))
    bound = server.server_address[:2]
    print(f"[easydetect serve] {model} ({task}) on http://{bound[0]}:{bound[1]} — "
          f"POST /predict, GET /health; Ctrl+C stops")
    if ready is not None:
        ready(bound)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
