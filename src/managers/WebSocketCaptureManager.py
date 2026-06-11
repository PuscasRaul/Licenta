#!/usr/bin/env python3
import threading

import cv2 as cv
import numpy as np

from src.managers.ICaptureManager import ICaptureManager
from src.domain.Frame import Frame


class WebSocketCaptureManager(ICaptureManager):
    '''
    Demo-application implementation of ICaptureManager: the capture source is
    not a camera attached to the device but a live video feed delivered over
    a WebSocket connection. Encoded frames arrive continuously from an
    external client, are decoded and wrapped in the same Frame object,
    leaving the interface contract unchanged.

    As in the embedded case, only the most recently received frame is kept
    available, avoiding latency accumulation in real-time processing.
    Substituting the physical source with a network feed, without touching
    the processing flow, demonstrates the benefit of the interface
    abstraction: the core stays indifferent to the nature of the source.
    '''

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_count = 0

    def start(self) -> None:
        pass

    def submit(self, encoded_bytes) -> None:
        '''
        Decode an encoded frame received over the socket and retain it as the
        latest available frame.
        '''
        buffer = np.frombuffer(encoded_bytes, dtype=np.uint8)
        image = cv.imdecode(buffer, cv.IMREAD_GRAYSCALE)
        if image is None:
            return
        with self._lock:
            self._latest_frame = image
            self._frame_count += 1

    def get_latest_frame(self) -> Frame:
        with self._lock:
            image = self._latest_frame
            count = self._frame_count
        if image is None:
            return None
        metadata = {"source": "websocket", "frame_index": count}
        return Frame(image, None, metadata)

    def stop(self) -> None:
        with self._lock:
            self._latest_frame = None
