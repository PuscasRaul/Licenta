#!/usr/bin/env python3
import cv2 as cv
from ICaptureManager import ICaptureManager
from src.domain.Frame import Frame


class CameraCaptureManager(ICaptureManager):
    def __init__(self, device_index=0):
        self._device_index = device_index
        self._video_capture = cv.VideoCapture(self._device_index)

        if not self._video_capture.isOpened():
            raise ValueError(f"Could not open video source: {device_index}")

        for i in range(0, 10):  # To determine widht and height accurately
            success, frame = self._video_capture.read()

        self._fps = self._video_capture.get(cv.CAP_PROP_FPS)  # inaccurate
        self._size = frame.shape[:2]
        self._video_capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

    '''
    def _reader(self):
        TODO: Check which solution is better for getting the latest frame
        import threading
        self._lock = threading.Lock()
        self._read_thread = threading.Thread(target=self._reader)
        self._read_thread.daemon = True
        while (True):
            with self._lock:
                ret = self._video_capture.grab()
            if not ret:
                break
            # Possibly, add a sleep of (1/self._fps so the other thr gets time)
            self._read_thread.sleep(1/self._fps)  # WARN: maybe unnecessary
    '''

    def start(self) -> None:
        self._read_thread.start()
        pass

    def get_latest_frame(self) -> Frame:
        with self._lock:
            _, frame = self._video_capture.retrieve()
        success, frame = self._video_capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        metadata = ""  # should be a dictionary containing all important info
        return Frame(
            frame,
            self._video_capture.get(cv.CAP_PROP_POS_MSEC + 1000/self._fps),
            metadata)

    def stop(self) -> None:
        self._video_capture.release()
