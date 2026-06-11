#!/usr/bin/env python3
import cv2 as cv
import threading
from src.managers.ICaptureManager import ICaptureManager
from src.domain.Frame import Frame


class CameraCaptureManager(ICaptureManager):
    '''
    Embedded-system implementation of ICaptureManager: pulls frames from a
    camera attached to the device, identified by its device index.

    A dedicated reader thread continuously grabs frames and keeps only the
    most recent one available. This avoids the cumulative latency that
    sequential reads would introduce, since capture backends buffer frames
    internally. Concurrent access to the shared frame is guarded by a lock.
    '''

    def __init__(self, device_index=0):
        self._device_index = device_index
        self._video_capture = cv.VideoCapture(self._device_index)

        if not self._video_capture.isOpened():
            raise ValueError(f"Could not open video source: {device_index}")

        for _ in range(0, 10):  # warm-up to determine width and height
            success, frame = self._video_capture.read()

        self._fps = self._video_capture.get(cv.CAP_PROP_FPS)  # inaccurate
        self._size = frame.shape[:2]
        self._video_capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._latest_frame = frame
        self._running = False
        self._read_thread = threading.Thread(target=self._reader, daemon=True)

    def _reader(self) -> None:
        '''
        Continuously read frames, retaining only the latest one. Runs on a
        background thread so the processing core always gets a fresh frame.
        '''
        while self._running:
            success, frame = self._video_capture.read()
            if not success:
                break
            with self._lock:
                self._latest_frame = frame

    def start(self) -> None:
        self._running = True
        self._read_thread.start()

    def get_latest_frame(self) -> Frame:
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return None

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        timestamp = self._video_capture.get(cv.CAP_PROP_POS_MSEC)
        metadata = {"source": "camera", "device_index": self._device_index}
        return Frame(frame, timestamp, metadata)

    def stop(self) -> None:
        self._running = False
        if self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        self._video_capture.release()
