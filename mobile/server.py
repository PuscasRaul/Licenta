#!/usr/bin/env python3
'''
Lightweight, phone-friendly demo edge for the IORA system.

It serves the single-page mobile client and accepts a live camera feed over a
WebSocket, feeding it through the *unchanged* processing core
(localize -> segment -> recognize) and streaming the recognised plate back.

Unlike src/main.py (FastAPI/uvicorn), this entry point depends only on the
pure-Python ``websockets`` library plus the runtime stack already required by
the pipeline (OpenCV, NumPy, scikit-learn, joblib). That avoids the
FastAPI/pydantic/uvicorn build chain, which is awkward to compile on Android
(Termux). The processing core and both interface edges
(WebSocketCaptureManager, WebSocketApplicationLogic) are reused as-is, which is
exactly what the swappable-edge architecture is meant to allow.

Run from the project root:

    python -m mobile.server            # or:  python mobile/server.py

then open http://localhost:8000 in the phone browser.
'''
import asyncio
import http
import json
import os
import sys

# Allow ``import src.*`` when launched as a plain script (python mobile/server.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from src.domain.Frame import Frame
from src.pipeline.LPExtraction import LPExtraction
from src.pipeline.CharacterSegmentation import CharacterSegmentation
from src.pipeline.CharacterRecognition import CharacterRecognition
from src.pipeline.ProcessingPipeline import ProcessingPipeline
from src.managers.WebSocketCaptureManager import WebSocketCaptureManager
from src.application.WebSocketApplicationLogic import WebSocketApplicationLogic

HOST = os.environ.get("IORA_HOST", "0.0.0.0")
PORT = int(os.environ.get("IORA_PORT", "8000"))

# Centred region-of-interest kept from each incoming frame before it reaches the
# pipeline. Whatever client is connected (the browser page here or the Ionic app
# in mobileapp/), the user aims the plate at the middle of the camera and the
# surrounding clutter -- the rest of a laptop screen, the desk, the room, seen at
# an angle -- is discarded so the localizer is not distracted by it. Fractions of
# 1.0 disable the crop; override with IORA_CROP_W / IORA_CROP_H.
CROP_W_FRAC = float(os.environ.get("IORA_CROP_W", "0.8"))
CROP_H_FRAC = float(os.environ.get("IORA_CROP_H", "0.5"))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "svm_model.joblib")
DATASET_DIRS = [os.path.join(_PROJECT_ROOT, "data", "labeled")]
_STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_STATIC_DIR, "index.html")

# Static assets the client page pulls in, mapped to their MIME type. The page
# used to be a single inline file; the stylesheet and script now live beside it
# and are fetched separately, so the server has to hand them out too.
_STATIC_FILES = {
    "/style.css": "text/css; charset=utf-8",
    "/script.js": "text/javascript; charset=utf-8",
}

PIPELINE = None


def build_pipeline():
    '''Load the persisted SVM classifier (or train one if absent) and wire the
    three-stage processing core, mirroring src/main.py.'''
    global PIPELINE
    extraction = LPExtraction((1.5, 8.0), 500)
    segmentation = CharacterSegmentation()
    if os.path.isfile(MODEL_PATH):
        recognition = CharacterRecognition.load(MODEL_PATH)
    else:
        recognition = CharacterRecognition().train_from_dirs(DATASET_DIRS)
        recognition.save(MODEL_PATH)
    PIPELINE = ProcessingPipeline(extraction, segmentation, recognition)
    print("IORA pipeline ready.")


def _center_crop(image):
    '''Return the centred CROP_W_FRAC x CROP_H_FRAC sub-rectangle of a decoded
    frame. Returns the image unchanged when both fractions are >= 1.0 or it is
    too small to crop, so the rest of the pipeline never sees an empty array.'''
    if image is None or (CROP_W_FRAC >= 1.0 and CROP_H_FRAC >= 1.0):
        return image
    h, w = image.shape[:2]
    cw = max(1, min(w, int(round(w * CROP_W_FRAC))))
    ch = max(1, min(h, int(round(h * CROP_H_FRAC))))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return image[y0:y0 + ch, x0:x0 + cw]


class _JsonSocket:
    '''Adapter giving the ``websockets`` connection the ``send_json`` method
    that WebSocketApplicationLogic duck-types against, so that edge is reused
    unchanged.'''

    def __init__(self, ws) -> None:
        self._ws = ws

    async def send_json(self, obj) -> None:
        await self._ws.send(json.dumps(obj))


async def _handle_frames(websocket) -> None:
    '''One WebSocket client: ingest frames on one task, process the most recent
    frame on another so a slow inference never blocks ingestion (stale frames
    are simply dropped, since the capture manager keeps only the latest).'''
    capture = WebSocketCaptureManager()
    app_logic = WebSocketApplicationLogic(_JsonSocket(websocket))
    capture.start()
    stop = asyncio.Event()

    async def reader():
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    capture.submit(message)
        finally:
            stop.set()

    async def processor():
        last_index = -1
        while not stop.is_set():
            frame = capture.get_latest_frame()
            if frame is None or PIPELINE is None:
                await asyncio.sleep(0.05)
                continue
            index = (frame.get_metadata or {}).get("frame_index")
            if index == last_index:
                await asyncio.sleep(0.03)
                continue
            last_index = index
            cropped = _center_crop(frame.get_image_data)
            frame = Frame(cropped, frame.get_timestamp(), frame.get_metadata)
            result = await asyncio.to_thread(PIPELINE.process, frame)
            try:
                await app_logic.handle(result)
            except websockets.ConnectionClosed:
                break

    try:
        await asyncio.gather(reader(), processor())
    finally:
        capture.stop()


def _serve_file(path, content_type):
    with open(path, "rb") as handle:
        body = handle.read()
    headers = Headers([
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ])
    return Response(200, "OK", headers, body)


async def _router(websocket):
    '''Frames arrive on /ws; every other path is handled in process_request.'''
    await _handle_frames(websocket)


def _process_request(connection, request):
    '''Serve the mobile client for ordinary GET requests; let /ws upgrade to a
    WebSocket (return None) so the same port handles both.'''
    if request.path == "/ws":
        return None
    if request.path in ("/", "/index.html"):
        return _serve_file(_INDEX_HTML, "text/html; charset=utf-8")
    if request.path in _STATIC_FILES:
        return _serve_file(os.path.join(_STATIC_DIR, request.path.lstrip("/")),
                           _STATIC_FILES[request.path])
    return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")


async def main():
    build_pipeline()
    async with serve(_router, HOST, PORT, process_request=_process_request,
                      max_size=8 * 1024 * 1024):
        print(f"IORA mobile demo on http://localhost:{PORT}  "
              f"(serving {HOST}:{PORT})")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")
