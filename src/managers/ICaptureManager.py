#!/usr/bin/env python3
from src.domain.Frame import Frame


class ICaptureManager(object):
    '''
    Abstract contract for the input module. Hides the physical nature of
    the capture source (camera, network feed, ...) behind three methods so
    the processing core only ever sees Frame objects.
    '''

    def start(self) -> None:
        pass

    def get_latest_frame(self) -> Frame:
        pass

    def stop(self) -> None:
        pass
