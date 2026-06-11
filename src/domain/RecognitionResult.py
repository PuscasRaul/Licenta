#!/usr/bin/env python3


class RecognitionResult(object):
    '''
    Output of the processing pipeline, handed to the application-logic
    module. A minimal data structure independent of any external library:
    the recognized plate string, the timestamp of the originating frame and
    an optional metadata bag carried over from the input Frame.
    '''

    def __init__(self, plate, timestamp=None, metadata=None) -> None:
        self._plate = plate
        self._timestamp = timestamp
        self._metadata = metadata if metadata is not None else {}

    @property
    def plate(self) -> str:
        return self._plate

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __repr__(self) -> str:
        return f"RecognitionResult(plate={self._plate!r})"
