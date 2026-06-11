#!/usr/bin/env python3
from src.application.IApplicationLogic import IApplicationLogic
from src.domain.RecognitionResult import RecognitionResult


class AccessControlApplicationLogic(IApplicationLogic):
    '''
    Embedded-system implementation: an access-control checkpoint. Recognizing
    a plate present in an authorization list raises a barrier by actuating
    the corresponding hardware; in parallel, a notification can be emitted
    (e.g. by e-mail) for both authorized accesses and unauthorized attempts.

    The barrier actuation and the notification are injected as callables so
    that this class stays independent of any concrete hardware or mailing
    library, keeping the dependency at the edge of the system.
    '''

    def __init__(self, authorized_plates, raise_barrier=None,
                 notify=None) -> None:
        self._authorized = {p.upper() for p in authorized_plates}
        self._raise_barrier = raise_barrier or self._default_raise_barrier
        self._notify = notify or self._default_notify

    def handle(self, result: RecognitionResult) -> None:
        if result is None or not result.plate:
            return

        plate = result.plate.upper()
        authorized = plate in self._authorized

        if authorized:
            self._raise_barrier(plate)
            self._notify(plate, authorized=True)
        else:
            self._notify(plate, authorized=False)

    @staticmethod
    def _default_raise_barrier(plate) -> None:
        print(f"[barrier] access granted, raising barrier for {plate}")

    @staticmethod
    def _default_notify(plate, authorized) -> None:
        status = "authorized" if authorized else "unauthorized"
        print(f"[notify] {status} access attempt: {plate}")
