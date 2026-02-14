#!/usr/bin/env python3
import cv2 as cv
import threading

class Frame(object):
    def __init__(self, image_data, timestamp, metadata) -> None:
        self._image_data = image_data
        self._timestap = timestamp
        self._metadata = metadata

    @property
    def get_metadata(self) -> str:
        return self._metadata

    @property
    def get_image_data(self):
        return self._image_data

    def get_timestamp(self):
        return self._timestap


class ICaptureManger(object):
    def start(self) -> None:
        pass

    def get_latest_frame(self) -> Frame:
        pass

    def stop(self) -> None:
        pass


class CvCaptureManager(ICaptureManger):
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
        TODO: Check which solution is better for getting the latest frame
        self._lock = threading.Lock()
        self._read_thread = threading.Thread(target=self._reader)
        self._read_thread.daemon = True
        '''

    '''
    def _reader(self):
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
        metadata = ""  # should be a dictionary containing all important info
        return Frame(
            frame,
            self._video_capture.get(cv.CAP_PROP_POS_MSEC + 1000/self._fps),
            metadata)

    def stop(self) -> None:
        pass
