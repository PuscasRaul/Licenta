#!/usr/bin/env python3
from src.domain.RecognitionResult import RecognitionResult


class IApplicationLogic(object):
    '''
    Abstract contract for the application-logic module, one of the two
    extensible edges of the system. It consumes the result produced by the
    processing pipeline and triggers the action appropriate to the execution
    context, keeping the core neutral to the concrete purpose of the app.
    '''

    def handle(self, result: RecognitionResult) -> None:
        pass
