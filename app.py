"""
FastAPI app for meeting-recorder.

State model: one recording at a time, in-memory only.
WebSocket clients receive JSON status messages as the pipeline progresses.
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
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

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("obsws_python").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

# "idle" | "recording" | "stopped" | "transcribing" | "done" | "error"
_state: str = "idle"
_ws_clients: Set[WebSocket] = set()
_stopped_path: str | None = None  # OBS output path held between stop and user naming
_recording_started_at: float | None = None


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

@app.post("/reset")
async def reset_state():
    """Reset state to idle — use to recover from a stuck error state."""
    await _send_status("idle")
    return {"state": "idle"}


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
        }
        for key, cfg in config.MODELS.items()
    ]


@app.post("/recording/start")
async def recording_start():
    global _recording_started_at
    try:
        obs.start_recording()
    except RuntimeError as e:
        # OBS not connected
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to start recording")
        return JSONResponse(status_code=503, content={"error": f"OBS error: {e}"})

    _recording_started_at = time.time()
    await _send_status("recording", "Recording started")
    return {"state": "recording"}


@app.post("/recording/stop")
async def recording_stop():
    global _stopped_path, _recording_started_at
    try:
        audio_path = obs.stop_recording()
    except RuntimeError as e:
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

    safe_name = (file.filename or "upload").replace(" ", "_")
    dest = recordings_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)

    recording_name = Path(safe_name).stem
    # Avoid clobbering an existing transcript
    if (Path(config.TRANSCRIPTS_DIR) / f"{recording_name}.txt").exists():
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
            summaries_dir = Path(config.SUMMARIES_DIR)
            summaries_dir.mkdir(parents=True, exist_ok=True)
            (summaries_dir / f"{name}.txt").write_text(summary, encoding="utf-8")
        return {"summary": summary}
    except Exception as e:
        logger.exception("Summarization failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/transcripts/{transcript_id}")
async def update_transcript(transcript_id: str, body: dict):
    """Save speaker name mappings to the transcript meta. Transcript text is never modified."""
    txt = Path(config.TRANSCRIPTS_DIR) / f"{transcript_id}.txt"
    if not txt.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    speakers = body.get("speakers")
    if not isinstance(speakers, dict):
        return JSONResponse(status_code=400, content={"error": "speakers dict required"})
    json_path = Path(config.TRANSCRIPTS_DIR) / f"{transcript_id}.json"
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    meta["speakers"] = speakers
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    return {"updated": transcript_id}


@app.post("/export")
async def export_transcript(body: dict):
    destination = (body.get("destination") or "").strip()
    title = (body.get("title") or "Untitled Recording").strip()
    summary = (body.get("summary") or "").strip()
    transcript_id = (body.get("id") or "").strip()

    if transcript_id:
        # Load original transcript and apply current speaker substitution server-side
        txt_path = Path(config.TRANSCRIPTS_DIR) / f"{transcript_id}.txt"
        if not txt_path.exists():
            return JSONResponse(status_code=404, content={"error": f"Transcript not found: {transcript_id}"})
        original = txt_path.read_text(encoding="utf-8")
        json_path = Path(config.TRANSCRIPTS_DIR) / f"{transcript_id}.json"
        meta = json.loads(json_path.read_text()) if json_path.exists() else {}
        speakers = meta.get("speakers", {})
        transcript = original
        for tag, name in speakers.items():
            if name.strip():
                transcript = transcript.replace(tag, name.strip())
        # Load saved summary if not provided in body
        if not summary:
            summary_path = Path(config.SUMMARIES_DIR) / f"{transcript_id}.txt"
            if summary_path.exists():
                summary = summary_path.read_text(encoding="utf-8")
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
    recording_name = Path(audio_path).stem
    await _send_status("transcribing", "Processing recording...")
    asyncio.create_task(_run_pipeline(audio_path, recording_name, model_key, task))
    return {"state": "transcribing", "id": recording_name}


@app.get("/transcripts")
async def list_transcripts():
    """Return past transcripts, sorted newest first."""
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    summaries_dir = Path(config.SUMMARIES_DIR)
    if not transcripts_dir.exists():
        return []
    results = []
    for json_path in transcripts_dir.glob("*.json"):
        stem = json_path.stem
        txt_path = transcripts_dir / f"{stem}.txt"
        if not txt_path.exists() or txt_path.stat().st_size == 0:
            continue
        meta = json.loads(json_path.read_text())
        has_summary = (summaries_dir / f"{stem}.txt").exists()
        results.append({"id": stem, **meta, "has_summary": has_summary})
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results


@app.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str):
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    txt_path = transcripts_dir / f"{transcript_id}.txt"
    if not txt_path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    text = txt_path.read_text(encoding="utf-8")
    json_path = transcripts_dir / f"{transcript_id}.json"
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    result: dict = {"text": text, "meta": meta}
    if "speakers" in meta:
        result["speakers"] = meta["speakers"]
    if "speakers_list" in meta:
        result["speakers_list"] = meta["speakers_list"]
    summary_path = Path(config.SUMMARIES_DIR) / f"{transcript_id}.txt"
    if summary_path.exists():
        result["summary"] = summary_path.read_text(encoding="utf-8")
    return result


@app.delete("/transcripts/{transcript_id}")
async def delete_transcript(transcript_id: str):
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    txt_path = transcripts_dir / f"{transcript_id}.txt"
    if not txt_path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    txt_path.unlink(missing_ok=True)
    (transcripts_dir / f"{transcript_id}.json").unlink(missing_ok=True)
    (Path(config.SUMMARIES_DIR) / f"{transcript_id}.txt").unlink(missing_ok=True)
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

    # Write metadata alongside the transcript
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": Path(audio_path).name,
        "model": _model_label(model_key),
        "created_at": datetime.now().isoformat(),
    }
    (transcripts_dir / f"{recording_name}.json").write_text(json.dumps(meta))

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

        # Save the full speaker list to meta so the UI can always show all SPEAKER_XX inputs
        speakers_list = sorted(set(re.findall(r'SPEAKER_\d+', transcript)))
        meta['speakers_list'] = speakers_list
        (transcripts_dir / f"{recording_name}.json").write_text(json.dumps(meta))

        await _send_status("done", "Transcription complete")
        await _broadcast({"type": "transcript", "id": recording_name, "text": transcript, "model": model_label, "speakers_list": speakers_list})

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
