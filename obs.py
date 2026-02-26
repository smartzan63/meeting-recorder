import logging
import os
import time

import obsws_python as obs

import config

logger = logging.getLogger(__name__)

_client: obs.ReqClient | None = None


def connect() -> None:
    global _client
    _client = obs.ReqClient(
        host=config.OBS_HOST,
        port=config.OBS_PORT,
        password=config.OBS_PASSWORD,
        timeout=5,
    )
    logger.info("Connected to OBS at %s:%s", config.OBS_HOST, config.OBS_PORT)


def disconnect() -> None:
    global _client
    if _client is not None:
        try:
            _client.disconnect()
        except Exception:
            pass
        _client = None


def _require_client() -> obs.ReqClient:
    if _client is None:
        raise RuntimeError("Not connected to OBS")
    return _client


def start_recording() -> None:
    _require_client().start_record()
    logger.info("OBS recording started")


def _wait_for_file(path: str, timeout: float = 30.0, stable_secs: float = 0.5) -> None:
    """Block until path exists on disk and its size has stopped growing.

    OBS sends output_path in the StopRecord response before it finishes
    muxing the container, so the file may not be readable yet.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            size1 = os.path.getsize(path)
            time.sleep(stable_secs)
            if os.path.getsize(path) == size1:
                return
        else:
            time.sleep(0.2)
    raise RuntimeError(f"OBS file not ready after {timeout}s: {path}")


def stop_recording() -> str:
    """Stop OBS recording and return path to the saved file.

    obsws_python's stop_record() returns a StopRecord response whose
    outputPath field (snake_cased to output_path) contains the exact
    filesystem path OBS wrote the file to.
    """
    client = _require_client()
    resp = client.stop_record()
    obs_path: str = resp.output_path
    logger.info("OBS recording stopped, file reported at: %s", obs_path)

    # OBS reports the path as seen from the host OS. When running inside Docker
    # the host path is not accessible — remap to config.RECORDINGS_DIR using
    # just the filename, which lands in the volume-mounted recordings directory.
    import os as _os
    filename = _os.path.basename(obs_path)
    output_path = _os.path.join(config.RECORDINGS_DIR, filename)
    if output_path != obs_path:
        logger.info("OBS path remapped to: %s", output_path)

    _wait_for_file(output_path)
    logger.info("OBS file ready: %s", output_path)
    return output_path


def get_record_directory() -> str:
    """Return the directory OBS writes recordings into."""
    client = _require_client()
    resp = client.get_record_directory()
    return resp.record_directory
