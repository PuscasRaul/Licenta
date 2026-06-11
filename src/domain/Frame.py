#!/usr/bin/env python3

class Frame(object):
    def __init__(
            self,
            image_data,
            timestamp=None,
            metadata=None
    ) -> None:
        self._image_data = image_data
        self._timestamp = timestamp
        self._metadata = metadata

    @property
    def get_metadata(self) -> str:
        return self._metadata

    @property
    def get_image_data(self):
        return self._image_data

    def get_timestamp(self):
        return self._timestamp
