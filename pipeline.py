"""
Cloud transcription pipeline: ffmpeg (MKV only) → AI provider → transcript.

Provider is selected via PROVIDER env var: "gemini" (default) or "azure".
All heavy steps run in a thread pool executor to avoid blocking the event loop.
"""

import asyncio
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

import config

logger = logging.getLogger(__name__)

_MIME_MAP = {
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def _transcribe_with_gemini(
    audio_path: str,
    model_key: str,
    status_callback: Callable[[str], None],
) -> str:
    """Upload audio to Gemini Files API and return a speaker-labeled transcript."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set — add it to your .env file")

    from google import genai  # type: ignore
    from google.genai import types as _gtypes  # type: ignore

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    model_cfg = config.MODELS[model_key]
    model = model_cfg["model"]
    label = model_cfg["label"]
    mime_type = _MIME_MAP.get(Path(audio_path).suffix.lower(), "audio/wav")

    status_callback("Uploading audio to Gemini...")
    t0 = time.time()
    audio_file = client.files.upload(
        file=audio_path,
        config=_gtypes.UploadFileConfig(mime_type=mime_type, display_name=Path(audio_path).name),
    )
    logger.info("Gemini: uploaded %s in %.1fs → %s", audio_path, time.time() - t0, audio_file.uri)

    status_callback(f"Transcribing with {label}...")
    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=[
            """Transcribe this audio. Identify different speakers and label them SPEAKER_00, SPEAKER_01, etc.
Format the output exactly like this — each speaker turn on its own block:

[MM:SS] SPEAKER_00
text here

[MM:SS] SPEAKER_01
text here

Use timestamps (MM:SS) relative to the start of the audio. Keep the original language of each speaker.""",
            audio_file,
        ],
    )
    logger.info("Gemini: transcription done in %.1fs", time.time() - t0)

    try:
        client.files.delete(name=audio_file.name)
    except Exception:
        pass

    return response.text


def _transcribe_with_azure(
    audio_path: str,
    status_callback: Callable[[str], None],
) -> str:
    """Transcribe audio using Azure AI Speech Fast Transcription API with diarization."""
    import json
    import requests

    if not config.AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY is not set — add it to your .env file")

    region = config.AZURE_SPEECH_REGION
    url = f"https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

    definition = {
        "locales": ["en-US"],
        "diarization": {"enabled": True},
        "profanityFilterMode": "None",
    }

    status_callback("Uploading audio to Azure Speech...")
    t0 = time.time()

    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY},
            files={
                "audio": (Path(audio_path).name, audio_file, "audio/wav"),
                "definition": (None, json.dumps(definition), "application/json"),
            },
        )

    if response.status_code != 200:
        raise RuntimeError(f"Azure Speech API error {response.status_code}: {response.text}")

    logger.info("Azure Speech: transcription done in %.1fs", time.time() - t0)

    data = response.json()
    return _format_azure_transcript(data)


def _format_azure_transcript(data: dict) -> str:
    """Convert Azure Speech Fast Transcription response to SPEAKER_XX / [MM:SS] format."""
    lines = []
    for phrase in data.get("phrases", []):
        offset_ms = phrase.get("offsetMilliseconds", 0)
        speaker = phrase.get("speaker", 0)
        text = phrase.get("text", "").strip()
        if not text:
            continue
        mm = offset_ms // 60000
        ss = (offset_ms % 60000) // 1000
        speaker_label = f"SPEAKER_{int(speaker):02d}"
        lines.append(f"[{mm:02d}:{ss:02d}] {speaker_label}\n{text}")
    return "\n\n".join(lines)


def summarize_transcript(transcript: str) -> str:
    """Summarize a transcript using the configured provider. Returns markdown."""
    if config.PROVIDER == "azure":
        return _summarize_with_azure(transcript)
    return _summarize_with_gemini(transcript)


def _summarize_with_gemini(transcript: str) -> str:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai  # type: ignore

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.MODELS["gemini"]["model"],
        contents=[_summary_prompt(transcript)],
    )
    return response.text


def _summarize_with_azure(transcript: str) -> str:
    if not config.AZURE_OPENAI_KEY:
        raise RuntimeError("AZURE_OPENAI_KEY is not set — add it to your .env file")

    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version="2025-01-01-preview",
    )
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": _summary_prompt(transcript)}],
    )
    return response.choices[0].message.content


def _summary_prompt(transcript: str) -> str:
    return f"""Summarize this meeting transcript. Be concise and structured.
Include:
- Key topics discussed
- Decisions made (if any)
- Action items (if any)

Transcript:
{transcript}"""


def _run_pipeline_sync(
    audio_path: str,
    output_dir: str,
    status_callback: Callable[[str], None],
    model_key: str = config.DEFAULT_MODEL,
    task: str = "transcribe",
    wav_dir: str | None = None,
) -> str:
    """Synchronous cloud pipeline — intended to run in a thread pool executor.

    wav_dir: if provided and input is MKV, the converted WAV is saved here
    (e.g. recordings/) before upload so it can be re-processed on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    # MKV always needs conversion; Azure Speech also works best with WAV
    ext = Path(audio_path).suffix.lower()
    if ext == ".mkv":
        status_callback("Converting MKV to WAV...")
        if wav_dir:
            os.makedirs(wav_dir, exist_ok=True)
            stem = Path(audio_path).stem[:30].replace(" ", "_")
            wav_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}.wav"
            upload_path = os.path.join(wav_dir, wav_name)
        else:
            upload_path = os.path.join(output_dir, f"{run_id}_audio.wav")
        _convert_to_wav(audio_path, upload_path)
        logger.info("WAV ready: %s", upload_path)
        status_callback(f"Audio saved: recordings/{os.path.basename(upload_path)}")
    else:
        upload_path = audio_path

    if config.PROVIDER == "azure":
        transcript = _transcribe_with_azure(upload_path, status_callback)
    else:
        transcript = _transcribe_with_gemini(upload_path, model_key, status_callback)

    transcript_path = os.path.join(output_dir, f"{run_id}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    logger.info("Transcript saved: %s", transcript_path)

    status_callback("Pipeline complete.")
    return transcript


async def run(audio_path: str, output_dir: str, status_callback: Callable[[str], None]) -> str:
    """Run the cloud pipeline asynchronously."""
    loop = asyncio.get_running_loop()

    def sync_callback(message: str) -> None:
        loop.call_soon_threadsafe(status_callback, message)

    return await loop.run_in_executor(
        None,
        lambda: _run_pipeline_sync(audio_path, output_dir, sync_callback),
    )
