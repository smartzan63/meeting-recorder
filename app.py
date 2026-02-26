"""
FastAPI app for meeting-recorder.

State model: one recording at a time, in-memory only.
WebSocket clients receive JSON status messages as the pipeline progresses.
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Set

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

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

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("obsws_python").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

# "idle" | "recording" | "stopped" | "transcribing" | "done" | "error"
_state: str = "idle"
_ws_clients: Set[WebSocket] = set()
_stopped_path: str | None = None  # OBS output path held between stop and user naming


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
    await _broadcast({"type": "status", "state": state, "message": message})


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to OBS on startup; log a warning if it is not running
    try:
        obs.connect()
    except Exception as e:
        logger.warning("Could not connect to OBS on startup: %s", e)

    yield
    obs.disconnect()


app = FastAPI(lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/models")
async def models():
    if config.PROVIDER == "azure":
        return [{"key": "azure", "label": "Azure AI Speech", "default": True}]
    return [
        {
            "key": key,
            "label": cfg["label"],
            "default": key == config.DEFAULT_MODEL,
        }
        for key, cfg in config.MODELS.items()
    ]


@app.post("/recording/start")
async def recording_start():
    try:
        obs.start_recording()
    except RuntimeError as e:
        # OBS not connected
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to start recording")
        return JSONResponse(status_code=503, content={"error": f"OBS error: {e}"})

    await _send_status("recording", "Recording started")
    return {"state": "recording"}


@app.post("/recording/stop")
async def recording_stop():
    global _stopped_path
    try:
        audio_path = obs.stop_recording()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to stop recording")
        return JSONResponse(status_code=503, content={"error": f"OBS error: {e}"})

    _stopped_path = audio_path
    default_name = Path(audio_path).stem.replace(" ", "_")
    await _send_status("stopped", "")
    return {"state": "stopped", "default_name": default_name}


@app.post("/recording/save")
async def recording_save(body: dict):
    global _stopped_path
    if not _stopped_path:
        return JSONResponse(status_code=400, content={"error": "No stopped recording"})
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "Name is required"})

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
    return {"wav_path": wav_path}


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
    recordings_dir.mkdir(exist_ok=True)

    safe_name = (file.filename or "upload").replace(" ", "_")
    dest = recordings_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)

    recording_id = str(uuid.uuid4())
    await _send_status("transcribing", "Processing uploaded file...")
    asyncio.create_task(_run_pipeline(str(dest), recording_id, model, task))
    return {"state": "transcribing", "id": recording_id}


@app.post("/summarize")
async def summarize(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    loop = asyncio.get_running_loop()
    try:
        summary = await loop.run_in_executor(None, lambda: pipeline.summarize_transcript(text))
        return {"summary": summary}
    except Exception as e:
        logger.exception("Summarization failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


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
    recording_id = str(uuid.uuid4())
    await _send_status("transcribing", "Processing recording...")
    asyncio.create_task(_run_pipeline(audio_path, recording_id, model_key, task))
    return {"state": "transcribing", "id": recording_id}


@app.get("/transcripts")
async def list_transcripts():
    """Return past transcript runs, sorted newest first."""
    base = Path(config.TRANSCRIPTS_DIR)
    if not base.exists():
        return []
    results = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        txt_files = sorted(d.glob("*_transcript.txt"))
        if not txt_files or txt_files[0].stat().st_size == 0:
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            # Legacy folders: use folder mtime and wav stem if available
            wav_files = sorted(d.glob("*_audio.wav"))
            source = wav_files[0].stem.split("_", 1)[-1] if wav_files else d.name[:8]
            meta = {
                "source": source,
                "model": "",
                "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
            }
        results.append({"id": d.name, **meta})
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results


@app.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str):
    d = Path(config.TRANSCRIPTS_DIR) / transcript_id
    if not d.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    txt_files = sorted(d.glob("*_transcript.txt"))
    if not txt_files:
        return JSONResponse(status_code=404, content={"error": "No transcript file"})
    text = txt_files[0].read_text(encoding="utf-8")
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"text": text, "meta": meta}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    # Send current state immediately so the client can sync
    await ws.send_json({"type": "status", "state": _state, "message": ""})
    try:
        while True:
            # Keep the connection alive; we don't expect messages from the client
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_pipeline(audio_path: str, recording_id: str, model_key: str = config.DEFAULT_MODEL, task: str = "transcribe", save_wav: bool = False) -> None:
    logger.info("Pipeline started: source=%s model=%s task=%s", audio_path, model_key, task)
    output_dir = str(Path(config.TRANSCRIPTS_DIR) / recording_id)

    # Persist metadata so the history view can show a human-readable name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "source": Path(audio_path).name,
        "model": _model_label(model_key),
        "created_at": datetime.now().isoformat(),
    }
    (Path(output_dir) / "meta.json").write_text(json.dumps(meta))

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
                output_dir,
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
        await _send_status("done", "Transcription complete")
        await _broadcast({"type": "transcript", "id": recording_id, "text": transcript, "model": model_label})

    except Exception as e:
        logger.exception("Pipeline failed for recording %s", recording_id)
        status_queue.put_nowait(None)
        await drain_task
        await _send_status("error", f"Pipeline error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT, reload=False)
