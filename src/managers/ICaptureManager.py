#!/usr/bin/env python3
from src.domain import Frame


class ICaptureManger(object):
    def start(self) -> None:
        pass

    def get_latest_frame(self) -> Frame:
        pass

    def stop(self) -> None:
        pass
