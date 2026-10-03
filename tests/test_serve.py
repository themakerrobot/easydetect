# Apache-2.0
"""easydetect serve: pictures in over HTTP, the same detections out as JSON."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from .conftest import draw


@pytest.fixture(scope="module")
def server(tiny_ir):
    from easydetect import serve

    bound = {}
    ready = threading.Event()

    def on_ready(address):
        bound["url"] = f"http://{address[0]}:{address[1]}"
        ready.set()

    thread = threading.Thread(target=serve.serve, kwargs={
        "model": str(tiny_ir), "port": 0, "device": "CPU", "ready": on_ready}, daemon=True)
    thread.start()
    assert ready.wait(120), "the server did not come up"
    return bound["url"]


def _jpeg() -> bytes:
    import cv2

    return cv2.imencode(".jpg", draw())[1].tobytes()


def _post(url, body, content_type):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def test_health_names_the_model_and_its_classes(server):
    with urllib.request.urlopen(f"{server}/health", timeout=30) as response:
        info = json.loads(response.read())
    assert info["status"] == "ok" and info["names"] == {"0": "can", "1": "bottle"}


def test_a_raw_picture_gets_its_detections(server, tiny_ir):
    jpeg = _jpeg()
    status, kind, body = _post(f"{server}/predict?conf=0&max_det=3", jpeg, "image/jpeg")
    answer = json.loads(body)
    assert status == 200 and kind.startswith("application/json")
    assert answer["width"] == 80 and len(answer["detections"]) == 3
    assert {"name", "class", "confidence", "box"} <= set(answer["detections"][0])

    import cv2

    from easydetect import Detector

    # the same JPEG, decoded: it is lossy, and a random tiny model ranks its
    # boxes differently on the raw picture
    sent = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    same = Detector(str(tiny_ir), device="CPU", verbose=False)(sent, conf=0.0, max_det=3)[0]
    assert [d["box"] for d in answer["detections"]] == [d["box"] for d in same.summary()]


def test_a_form_upload_and_a_drawn_answer(server):
    boundary = "easydetectboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"p.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + _jpeg() + \
        f"\r\n--{boundary}--\r\n".encode()
    status, kind, jpg = _post(f"{server}/predict?conf=0&draw=1", body,
                              f"multipart/form-data; boundary={boundary}")
    assert status == 200 and kind == "image/jpeg" and jpg[:2] == b"\xff\xd8"


@pytest.mark.parametrize("body, content_type, path, code", [
    (b"not a picture", "image/jpeg", "/predict", 400),
    (b"", "image/jpeg", "/predict", 400),
    (b"x", "image/jpeg", "/elsewhere", 404),
])
def test_bad_requests_get_a_reason(server, body, content_type, path, code):
    with pytest.raises(urllib.error.HTTPError) as raised:
        _post(f"{server}{path}", body, content_type)
    assert raised.value.code == code and "error" in json.loads(raised.value.read())


def test_the_command_line_rejects_unknown_serve_keys():
    from easydetect import cli

    assert cli.main(["serve", "model=dfine-s", "conf=0.5"]) == 2
