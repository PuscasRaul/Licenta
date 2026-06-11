#!/usr/bin/env python3
from src.domain.RecognitionResult import RecognitionResult


class ProcessingPipeline():
    '''
    The processing flow, encapsulated as a self-contained unit with no public
    extension points. It consumes a Frame and applies, sequentially, the
    three recognition stages -- plate localization, character segmentation
    and character recognition -- producing a RecognitionResult that is handed
    to the application-logic module.
    '''

    def __init__(self, extraction, segmentation, recognition) -> None:
        self._extraction = extraction
        self._segmentation = segmentation
        self._recognition = recognition

    def process(self, frame) -> RecognitionResult:
        if frame is None:
            return None

        image = frame.get_image_data
        lps = self._extraction.extraction_pipeline(image)
        if lps is None or len(lps) == 0:
            return RecognitionResult("", frame.get_timestamp(),
                                     frame.get_metadata)

        characters = self._segmentation.character_segmentation(lps)
        if not characters:
            return RecognitionResult("", frame.get_timestamp(),
                                     frame.get_metadata)

        plate = self._recognition.predict(characters)
        return RecognitionResult(plate, frame.get_timestamp(),
                                 frame.get_metadata)
