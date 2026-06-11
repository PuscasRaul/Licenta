#!/usr/bin/env python3
from src.application.IApplicationLogic import IApplicationLogic
from src.domain.RecognitionResult import RecognitionResult


class WebSocketApplicationLogic(IApplicationLogic):
    '''
    Demo-application implementation: instead of acting on physical hardware,
    it exposes the recognition results to an external client over a
    persistent WebSocket connection. Each result is streamed in real time as
    it is produced by the processing pipeline.

    The send is asynchronous (the WebSocket I/O is awaited), so the demo edge
    overrides handle with an async coroutine. The websocket object is
    duck-typed (anything exposing async send_json) to avoid coupling this
    module to a specific web framework.
    '''

    def __init__(self, websocket) -> None:
        self._websocket = websocket

    async def handle(self, result: RecognitionResult) -> None:
        if result is None:
            return
        await self._websocket.send_json({
            "plate": result.plate,
            "timestamp": result.timestamp,
            "metadata": result.metadata,
        })
