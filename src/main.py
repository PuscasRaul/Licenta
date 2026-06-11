#!/usr/bin/env python3
'''
Demo application for the IORA system.

Wires the three modules together over a single WebSocket connection:
  - input edge:        WebSocketCaptureManager (frames arrive from the client)
  - processing core:   ProcessingPipeline (localize -> segment -> recognize)
  - application edge:   WebSocketApplicationLogic (results streamed back)

The same processing core is reused unchanged from the embedded scenario; only
the input and application edges are substituted with their WebSocket variants.
'''
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.pipeline.LPExtraction import LPExtraction
from src.pipeline.CharacterSegmentation import CharacterSegmentation
from src.pipeline.CharacterRecognition import CharacterRecognition
from src.pipeline.ProcessingPipeline import ProcessingPipeline
from src.managers.WebSocketCaptureManager import WebSocketCaptureManager
from src.application.WebSocketApplicationLogic import WebSocketApplicationLogic

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(project_root, 'data', 'svm_model.joblib')
DATASET_DIRS = [os.path.join(project_root, 'data', 'labeled')]

app = FastAPI()

_extraction = LPExtraction((1.5, 8.0), 500)
_segmentation = CharacterSegmentation()
_recognition = None
_pipeline = None


@app.on_event("startup")
def _build_pipeline() -> None:
    '''
    Load a persisted classifier if present, otherwise train one from the
    labeled dataset. If neither is available the demo still serves, but the
    pipeline is left disabled.
    '''
    global _recognition, _pipeline
    try:
        if os.path.isfile(MODEL_PATH):
            _recognition = CharacterRecognition.load(MODEL_PATH)
        else:
            _recognition = CharacterRecognition().train_from_dirs(DATASET_DIRS)
            _recognition.save(MODEL_PATH)
        _pipeline = ProcessingPipeline(_extraction, _segmentation,
                                       _recognition)
        print("IORA pipeline ready.")
    except Exception as exc:  # pragma: no cover - demo convenience
        print(f"IORA pipeline disabled: {exc}")


html = """
<!DOCTYPE html>
<html>
    <head><title>IORA demo</title></head>
    <body>
        <h1>IORA - recunoastere numere de inmatriculare</h1>
        <video id="video" autoplay playsinline width="480"></video>
        <ul id="results"></ul>
        <script>
            const ws = new WebSocket("ws://localhost:8000/ws");
            const video = document.getElementById("video");
            const canvas = document.createElement("canvas");
            navigator.mediaDevices.getUserMedia({video: true})
                .then(s => { video.srcObject = s; });
            ws.onmessage = e => {
                const data = JSON.parse(e.data);
                if (!data.plate) return;
                const li = document.createElement("li");
                li.textContent = data.plate;
                document.getElementById("results").appendChild(li);
            };
            setInterval(() => {
                if (ws.readyState !== 1 || !video.videoWidth) return;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext("2d").drawImage(video, 0, 0);
                canvas.toBlob(b => b.arrayBuffer().then(a => ws.send(a)),
                             "image/jpeg");
            }, 1000);
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    capture = WebSocketCaptureManager()
    app_logic = WebSocketApplicationLogic(websocket)
    capture.start()
    try:
        while True:
            data = await websocket.receive_bytes()
            capture.submit(data)
            frame = capture.get_latest_frame()
            if frame is None or _pipeline is None:
                continue
            result = _pipeline.process(frame)
            await app_logic.handle(result)
    except WebSocketDisconnect:
        capture.stop()
