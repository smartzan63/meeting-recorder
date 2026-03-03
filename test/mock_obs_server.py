"""
Mock OBS WebSocket server (obsws protocol v5).

Mimics enough of the OBS WebSocket v5 protocol for Level 1 testing.
Runs on port 4456 by default (configurable via OBS_PORT env var) to avoid
conflict with a real OBS instance on 4455.

Usage:
    python test/mock_obs_server.py

Env vars:
    OBS_PORT        WebSocket port to listen on (default: 4456)
    RECORDINGS_DIR  Directory where the mock WAV file is written (default: ./recordings)
"""

import asyncio
import json
import logging
import os
import struct
import time
from datetime import datetime

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mock-obs] %(message)s")
logger = logging.getLogger(__name__)

OBS_PORT = int(os.getenv("OBS_PORT", "4456"))
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "./recordings")


def make_minimal_wav() -> bytes:
    """Return a 44-byte minimal valid WAV header with no audio samples."""
    num_samples = 0
    num_channels = 1
    sample_rate = 16000
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', chunk_size, b'WAVE',
        b'fmt ', 16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )


def _success_response(request_type: str, request_id: str, output_data: dict | None = None) -> dict:
    """Build an obsws v5 RequestResponse (op=7) success message."""
    response: dict = {
        "op": 7,
        "d": {
            "requestType": request_type,
            "requestId": request_id,
            "requestStatus": {
                "result": True,
                "code": 100,
            },
        },
    }
    if output_data is not None:
        response["d"]["responseData"] = output_data
    return response


async def handle_connection(websocket) -> None:
    remote = websocket.remote_address
    logger.info("Client connected: %s", remote)

    # op=0 Hello — sent immediately on connect
    hello = {
        "op": 0,
        "d": {
            "obsWebSocketVersion": "5.0.0",
            "rpcVersion": 1,
            "authentication": None,
        },
    }
    await websocket.send(json.dumps(hello))
    logger.info("Sent Hello (op=0)")

    try:
        async for raw_message in websocket:
            logger.info("Received: %s", raw_message)
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message received, ignoring")
                continue

            op = msg.get("op")
            data = msg.get("d", {})

            if op == 1:
                # Identify — respond with Identified (op=2)
                identified = {
                    "op": 2,
                    "d": {"negotiatedRpcVersion": 1},
                }
                await websocket.send(json.dumps(identified))
                logger.info("Sent Identified (op=2)")

            elif op == 6:
                # Request (op=6)
                request_type = data.get("requestType", "")
                request_id = data.get("requestId", "")

                if request_type == "StartRecord":
                    logger.info("StartRecord received")
                    response = _success_response(request_type, request_id)
                    await websocket.send(json.dumps(response))
                    logger.info("Sent StartRecord success (op=7)")

                elif request_type == "StopRecord":
                    logger.info("StopRecord received — writing mock WAV")
                    os.makedirs(RECORDINGS_DIR, exist_ok=True)
                    filename = f"mock_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    wav_path = os.path.join(RECORDINGS_DIR, filename)
                    with open(wav_path, "wb") as f:
                        f.write(make_minimal_wav())
                    logger.info("Mock WAV written: %s", wav_path)

                    response = _success_response(
                        request_type,
                        request_id,
                        output_data={"outputPath": wav_path},
                    )
                    await websocket.send(json.dumps(response))
                    logger.info("Sent StopRecord success with path: %s", wav_path)

                else:
                    # Unknown request — respond with generic success
                    logger.warning("Unknown requestType: %s — sending generic success", request_type)
                    response = _success_response(request_type, request_id)
                    await websocket.send(json.dumps(response))

            else:
                logger.warning("Unhandled op: %s", op)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected cleanly: %s", remote)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning("Client disconnected with error: %s — %s", remote, e)
    except Exception as e:
        logger.exception("Unexpected error handling connection from %s: %s", remote, e)


async def main() -> None:
    logger.info("Starting mock OBS WebSocket server on port %d", OBS_PORT)
    logger.info("Recordings directory: %s", RECORDINGS_DIR)

    async with websockets.serve(handle_connection, "0.0.0.0", OBS_PORT):
        logger.info("Mock OBS server ready — waiting for connections")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
