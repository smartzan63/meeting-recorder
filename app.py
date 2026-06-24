"""
FastAPI app for meeting-recorder.

State model: one recording at a time, in-memory only.
WebSocket clients receive JSON status messages as the pipeline progresses.
"""

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config


def _model_label(model_key: str) -> str:
    if model_key == "azure":
        return "Azure AI Speech"
    return config.MODELS.get(model_key, {}).get("label", model_key)


def _valid_model_key(model_key: str) -> bool:
    if config.PROVIDER == "azure":
        return model_key == "azure"
    return model_key in config.MODELS


def _default_model_key() -> str:
    return "azure" if config.PROVIDER == "azure" else config.DEFAULT_MODEL
import obs
import pipeline
import storage

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("obsws_python").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

# "idle" | "recording" | "stopped" | "transcribing" | "done" | "error"
_state: str = "idle"
_ws_clients: Set[WebSocket] = set()
_stopped_path: str | None = None  # OBS output path held between stop and user naming
_recording_started_at: float | None = None
_obs_connected: bool = False


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def _broadcast(message: dict) -> None:
    dead: Set[WebSocket] = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


async def _send_status(state: str, message: str = "") -> None:
    global _state
    _state = state
    payload: dict = {"type": "status", "state": state, "message": message}
    if state == "recording" and _recording_started_at is not None:
        payload["started_at"] = _recording_started_at
    await _broadcast(payload)


# ── OBS background reconnect ──────────────────────────────────────────────────

async def _obs_reconnect_loop() -> None:
    global _obs_connected
    while True:
        await asyncio.sleep(10)
        if not obs.is_connected():
            try:
                obs.try_reconnect()
                logger.info("OBS reconnected")
                if not _obs_connected:
                    _obs_connected = True
                    await _broadcast({"type": "obs_status", "connected": True})
            except Exception as e:
                logger.debug("OBS reconnect attempt failed: %s", e)
                if _obs_connected:
                    _obs_connected = False
                    await _broadcast({"type": "obs_status", "connected": False})


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _obs_connected
    try:
        obs.connect()
        _obs_connected = True
    except Exception as e:
        logger.warning("Could not connect to OBS on startup: %s", e)
        _obs_connected = False

    reconnect_task = asyncio.create_task(_obs_reconnect_loop())
    yield
    reconnect_task.cancel()
    obs.disconnect()


app = FastAPI(lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset_state():
    """Reset state to idle — use to recover from a stuck error state."""
    await _send_status("idle")
    return {"state": "idle"}


@app.get("/obs/status")
async def obs_status():
    global _obs_connected
    loop = asyncio.get_running_loop()
    connected = await loop.run_in_executor(None, obs.is_connected)
    if connected != _obs_connected:
        _obs_connected = connected
        await _broadcast({"type": "obs_status", "connected": connected})
    return {"connected": connected}


@app.post("/obs/reconnect")
async def obs_reconnect():
    global _obs_connected
    try:
        obs.try_reconnect()
        if not _obs_connected:
            _obs_connected = True
            await _broadcast({"type": "obs_status", "connected": True})
        return {"connected": True}
    except Exception as e:
        if _obs_connected:
            _obs_connected = False
            await _broadcast({"type": "obs_status", "connected": False})
        return JSONResponse(status_code=503, content={"connected": False, "error": str(e)})


@app.get("/integrations")
async def integrations():
    return {
        "confluence": bool(config.CONFLUENCE_URL and config.CONFLUENCE_EMAIL and config.CONFLUENCE_TOKEN),
        "notion": bool(config.NOTION_TOKEN and config.NOTION_DATABASE_ID),
        "test_file_path": config.TEST_FILE_PATH or None,
    }


@app.get("/models")
async def models():
    if config.PROVIDER == "azure":
        return [{"key": "azure", "label": "Azure AI Speech", "default": True}]
    return [
        {
            "key": key,
            "label": cfg["label"],
            "default": key == config.DEFAULT_MODEL,
            "input_per_1m": cfg.get("input_per_1m"),
            "audio_per_1m": cfg.get("audio_per_1m"),
            "output_per_1m": cfg.get("output_per_1m"),
        }
        for key, cfg in config.MODELS.items()
    ]


@app.post("/recording/start")
async def recording_start():
    global _recording_started_at, _obs_connected
    try:
        obs.start_recording()
    except RuntimeError as e:
        if _obs_connected:
            _obs_connected = False
            await _broadcast({"type": "obs_status", "connected": False})
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to start recording")
        return JSONResponse(status_code=503, content={"error": f"OBS error: {e}"})

    _recording_started_at = time.time()
    await _send_status("recording", "Recording started")
    return {"state": "recording"}


@app.post("/recording/stop")
async def recording_stop():
    global _stopped_path, _recording_started_at, _obs_connected
    try:
        audio_path = obs.stop_recording()
    except RuntimeError as e:
        if _obs_connected:
            _obs_connected = False
            await _broadcast({"type": "obs_status", "connected": False})
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to stop recording")
        return JSONResponse(status_code=503, content={"error": f"OBS error: {e}"})

    _recording_started_at = None
    _stopped_path = audio_path
    default_name = Path(audio_path).stem.replace(" ", "_")
    await _send_status("stopped", "")
    return {"state": "stopped", "default_name": default_name}


@app.post("/recording/save")
async def recording_save(body: dict):
    global _stopped_path
    if not _stopped_path:
        return JSONResponse(status_code=400, content={"error": "No stopped recording"})
    raw_name = (body.get("name") or "").strip()
    if not raw_name:
        return JSONResponse(status_code=400, content={"error": "Name is required"})
    name = storage.sanitize_id(raw_name)

    recordings_dir = Path(config.RECORDINGS_DIR)
    recordings_dir.mkdir(exist_ok=True)
    wav_path = str(recordings_dir / f"{name}.wav")

    source = _stopped_path
    _stopped_path = None

    loop = asyncio.get_running_loop()
    import functools
    try:
        await loop.run_in_executor(
            None,
            functools.partial(pipeline._convert_to_wav, source, wav_path),
        )
    except Exception as e:
        logger.exception("WAV conversion failed")
        return JSONResponse(status_code=500, content={"error": f"Conversion failed: {e}"})

    logger.info("Recording saved as: %s", wav_path)
    await _send_status("idle", "")
    return {"wav_path": wav_path, "name": name}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form(default=config.DEFAULT_MODEL),
    task: str = Form(default="transcribe"),
):
    """Accept an audio file upload (M4A, WAV, MP4, MKV, …) and run the pipeline on it."""
    if _state == "transcribing":
        return JSONResponse(status_code=409, content={"error": "Already processing a file — wait for it to finish"})
    if not _valid_model_key(model):
        model = _default_model_key()
    if task not in ("transcribe", "translate"):
        task = "transcribe"

    recordings_dir = Path(config.RECORDINGS_DIR)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    orig = Path(file.filename or "upload")
    recording_name = storage.sanitize_id(orig.stem)
    safe_name = f"{recording_name}{orig.suffix.lower()}"
    dest = recordings_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)

    # Avoid clobbering an existing recording (storage is versioned dirs now).
    if storage.get_recording(recording_name) is not None:
        recording_name = f"{recording_name}_{int(time.time())}"

    await _send_status("transcribing", "Processing uploaded file...")
    asyncio.create_task(_run_pipeline(str(dest), recording_name, model, task))
    return {"state": "transcribing", "id": recording_name}


@app.post("/enrich")
async def enrich(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        return {"speakers": {}}
    loop = asyncio.get_running_loop()
    try:
        model_key = (body.get("model") or config.DEFAULT_MODEL).strip()
        if not _valid_model_key(model_key):
            model_key = _default_model_key()
        speakers = await loop.run_in_executor(None, lambda: pipeline.enrich_transcript(text, model_key))
        return {"speakers": speakers}
    except Exception as e:
        logger.warning("Enrichment failed (non-fatal): %s", e)
        return {"speakers": {}}


@app.post("/summarize")
async def summarize(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    name = (body.get("name") or "").strip()
    loop = asyncio.get_running_loop()
    try:
        model_key = (body.get("model") or config.DEFAULT_MODEL).strip()
        if not _valid_model_key(model_key):
            model_key = _default_model_key()
        summary = await loop.run_in_executor(None, lambda: pipeline.summarize_transcript(text, model_key))
        if name:
            storage.update_summary(name, summary)
        return {"summary": summary}
    except Exception as e:
        logger.exception("Summarization failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/transcripts/{transcript_id}")
async def update_transcript(transcript_id: str, body: dict):
    """Save speaker name mappings and/or summary against the active version."""
    if storage.get_recording(transcript_id) is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    speakers = body.get("speakers")
    if speakers is not None:
        if not isinstance(speakers, dict):
            return JSONResponse(status_code=400, content={"error": "speakers must be a dict"})
        storage.update_speakers(transcript_id, speakers)
    summary = body.get("summary")
    if summary is not None:
        storage.update_summary(transcript_id, summary)
    return {"updated": transcript_id}


@app.post("/export")
async def export_transcript(body: dict):
    destination = (body.get("destination") or "").strip()
    title = (body.get("title") or "Untitled Recording").strip()
    summary = (body.get("summary") or "").strip()
    transcript_id = (body.get("id") or "").strip()

    if transcript_id:
        rec = storage.get_recording(transcript_id)
        if rec is None:
            return JSONResponse(status_code=404, content={"error": f"Transcript not found: {transcript_id}"})
        transcript = rec["text"]
        for tag, name in (rec.get("speakers") or {}).items():
            if name.strip():
                transcript = transcript.replace(tag, name.strip())
        if not summary and rec.get("summary"):
            summary = rec["summary"]
    else:
        transcript = (body.get("transcript") or "").strip()

    if not transcript:
        return JSONResponse(status_code=400, content={"error": "No transcript provided"})

    if destination == "confluence":
        loop = asyncio.get_running_loop()
        try:
            url = await loop.run_in_executor(
                None, lambda: pipeline.export_to_confluence(title, transcript, summary)
            )
            return {"status": "ok", "url": url}
        except Exception as e:
            logger.exception("Confluence export failed")
            return JSONResponse(status_code=500, content={"error": str(e)})

    elif destination == "notion":
        loop = asyncio.get_running_loop()
        try:
            url = await loop.run_in_executor(
                None, lambda: pipeline.export_to_notion(title, transcript, summary)
            )
            return {"status": "ok", "url": url}
        except Exception as e:
            logger.exception("Notion export failed")
            return JSONResponse(status_code=500, content={"error": str(e)})

    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown destination: {destination}"})


@app.post("/test/process")
async def test_process(body: dict):
    """Dev-only: trigger pipeline on an existing file, bypassing OBS."""
    audio_path = body.get("file")
    if not audio_path or not Path(audio_path).exists():
        return JSONResponse(status_code=400, content={"error": f"File not found: {audio_path}"})
    model_key = body.get("model", _default_model_key())
    if not _valid_model_key(model_key):
        model_key = _default_model_key()
    task = body.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        task = "transcribe"
    recording_name = storage.sanitize_id(Path(audio_path).stem)
    await _send_status("transcribing", "Processing recording...")
    asyncio.create_task(_run_pipeline(audio_path, recording_name, model_key, task))
    return {"state": "transcribing", "id": recording_name}


@app.get("/transcripts")
async def list_transcripts():
    return storage.list_recordings()


@app.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str):
    rec = storage.get_recording(transcript_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    result: dict = {
        "text": rec["text"],
        "meta": {
            "source": rec.get("source", ""),
            "model": rec.get("model", ""),
            "created_at": rec.get("created_at", ""),
        },
        "active": rec["active"],
        "versions": rec["versions"],
    }
    if rec.get("speakers"):
        result["speakers"] = rec["speakers"]
    if rec.get("speakers_list"):
        result["speakers_list"] = rec["speakers_list"]
    if rec.get("summary"):
        result["summary"] = rec["summary"]
    return result


@app.post("/transcripts/{transcript_id}/active")
async def set_active_version(transcript_id: str, body: dict):
    """Switch which version is active (the one returned by GET /transcripts/{id})."""
    version_id = (body.get("version") or "").strip()
    if not version_id:
        return JSONResponse(status_code=400, content={"error": "version is required"})
    if not storage.set_active(transcript_id, version_id):
        return JSONResponse(status_code=404, content={"error": "Recording or version not found"})
    return {"id": transcript_id, "active": version_id}


@app.post("/transcripts/{transcript_id}/reprocess")
async def reprocess_transcript(transcript_id: str, body: dict):
    """Re-run transcription on the saved wav as a new version.

    Adds a version to the recording instead of overwriting; the new version
    becomes active. Switch back via POST /transcripts/{id}/active.
    """
    if _state == "transcribing":
        return JSONResponse(status_code=409, content={"error": "Already processing — wait for it to finish"})

    wav_path = Path(config.RECORDINGS_DIR) / f"{transcript_id}.wav"
    if not wav_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Original audio not found: {wav_path.name}"},
        )

    model_key = (body.get("model") or "").strip()
    if not _valid_model_key(model_key):
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model_key}"})

    task = body.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        task = "transcribe"

    await _send_status("transcribing", f"Reprocessing with {_model_label(model_key)}...")
    asyncio.create_task(_run_pipeline(str(wav_path), transcript_id, model_key, task))
    return {"state": "transcribing", "id": transcript_id}


@app.delete("/transcripts/{transcript_id}")
async def delete_transcript(transcript_id: str):
    if storage.get_recording(transcript_id) is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    storage.delete_recording(transcript_id)
    (Path(config.RECORDINGS_DIR) / f"{transcript_id}.wav").unlink(missing_ok=True)
    return {"deleted": transcript_id}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    # Send current state immediately so the client can sync
    init_payload: dict = {"type": "status", "state": _state, "message": ""}
    if _state == "recording" and _recording_started_at is not None:
        init_payload["started_at"] = _recording_started_at
    await ws.send_json(init_payload)
    await ws.send_json({"type": "obs_status", "connected": _obs_connected})
    try:
        while True:
            # Keep the connection alive; we don't expect messages from the client
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_pipeline(audio_path: str, recording_name: str, model_key: str = config.DEFAULT_MODEL, task: str = "transcribe", save_wav: bool = False) -> None:
    logger.info("Pipeline started: source=%s model=%s task=%s", audio_path, model_key, task)

    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    async def on_status(message: str) -> None:
        await _broadcast({"type": "status", "state": "transcribing", "message": message})

    # pipeline.run calls status_callback from a thread pool thread via
    # loop.call_soon_threadsafe, so we need a thread-safe wrapper here.
    # We use an asyncio.Queue to bridge the thread boundary cleanly.
    status_queue: asyncio.Queue[str | None] = asyncio.Queue()

    loop = asyncio.get_running_loop()

    def sync_status(message: str) -> None:
        # Called from executor thread — schedule onto the event loop captured above
        loop.call_soon_threadsafe(status_queue.put_nowait, message)

    # Drain the status queue in a background coroutine while pipeline runs
    async def drain_status():
        while True:
            msg = await status_queue.get()
            if msg is None:
                break
            await _broadcast({"type": "status", "state": "transcribing", "message": msg})

    drain_task = asyncio.create_task(drain_status())

    try:
        import functools
        transcript = await loop.run_in_executor(
            None,
            functools.partial(
                pipeline._run_pipeline_sync,
                audio_path,
                recording_name,
                sync_status,
                model_key,
                task,
                config.RECORDINGS_DIR if save_wav else None,
            ),
        )

        # Signal drain task to stop
        status_queue.put_nowait(None)
        await drain_task

        model_label = _model_label(model_key)

        # Pipeline writes a flat transcript file alongside its old contract; remove it
        # before recording the version, otherwise migrate_legacy would turn it into a
        # spurious extra version with an empty model field.
        (transcripts_dir / f"{recording_name}.txt").unlink(missing_ok=True)
        (transcripts_dir / f"{recording_name}.json").unlink(missing_ok=True)

        storage.add_version(
            recording_name,
            transcript,
            model_label,
            source=Path(audio_path).name,
        )

        speakers_list = sorted(set(re.findall(r'SPEAKER_\d+', transcript)))
        rec = storage.get_recording(recording_name)
        active_version = rec["active"] if rec else None
        versions = rec["versions"] if rec else []

        await _send_status("done", "Transcription complete")
        await _broadcast({
            "type": "transcript",
            "id": recording_name,
            "text": transcript,
            "model": model_label,
            "speakers_list": speakers_list,
            "active": active_version,
            "versions": versions,
        })

    except Exception as e:
        logger.exception("Pipeline failed for recording %s", recording_name)
        status_queue.put_nowait(None)
        await drain_task
        await _send_status("error", f"Pipeline error: {e}")


# ── Static file serving ───────────────────────────────────────────────────────

# Serve the React SPA — must be mounted last so API routes take priority
import os as _os
if _os.path.isdir("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse("frontend/dist/index.html")
else:
    # Fallback to old static HTML during development (before first build)
    @app.get("/{full_path:path}")
    async def serve_static(full_path: str):
        return FileResponse("static/index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT, reload=False)
