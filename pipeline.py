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


def enrich_transcript(transcript: str) -> dict[str, str]:
    """Identify real speaker names from transcript context. Returns a mapping like
    {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"} or {} if names cannot be determined.

    Only Azure OpenAI performs a real LLM call. All other providers return {} immediately.
    Never raises — returns {} on any error.
    """
    if config.PROVIDER != "azure":
        return {}

    try:
        if not config.AZURE_OPENAI_KEY:
            logger.warning("enrich_transcript: AZURE_OPENAI_KEY not set, skipping enrichment")
            return {}

        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_KEY,
            api_version="2025-01-01-preview",
        )

        prompt = (
            "You are analyzing a meeting transcript. Identify the real names of the speakers "
            "if they can be determined from the conversation context (e.g. someone is addressed "
            "by name, introduces themselves, or signs off with their name).\n\n"
            "Return ONLY a JSON object mapping speaker labels to names, e.g.:\n"
            '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}\n\n'
            "If a speaker's name cannot be determined, omit them from the result.\n"
            "If no names can be determined at all, return an empty object: {}\n\n"
            "Do not include any explanation, only the JSON object.\n\n"
            f"Transcript:\n{transcript}"
        )

        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = (response.choices[0].message.content or "").strip()

        import json
        result = json.loads(raw)
        if not isinstance(result, dict):
            return {}
        # Keep only string keys and string values
        return {k: v for k, v in result.items() if isinstance(k, str) and isinstance(v, str)}

    except Exception as e:
        logger.warning("enrich_transcript failed (non-fatal): %s", e)
        return {}


_MOCK_SUMMARY = """\
## Summary

**Key topics:** Test recording, speaker identification

**Decisions made:** None

**Action items:** None"""


def summarize_transcript(transcript: str) -> str:
    """Summarize a transcript using the configured provider. Returns markdown."""
    if config.PROVIDER == "mock":
        return _MOCK_SUMMARY
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


_MOCK_TRANSCRIPT = """\
[00:00] SPEAKER_00
This is a test recording.

[00:05] SPEAKER_01
Hello from speaker two. The quick brown fox.

[00:10] SPEAKER_00
Testing complete."""


def _run_pipeline_sync(
    audio_path: str,
    recording_name: str,
    status_callback: Callable[[str], None],
    model_key: str = config.DEFAULT_MODEL,
    task: str = "transcribe",
    wav_dir: str | None = None,
) -> str:
    """Synchronous cloud pipeline — intended to run in a thread pool executor.

    recording_name: stem name shared by transcript and summary files (no extension).
    wav_dir: if provided and input is MKV, the converted WAV is saved here.
    """
    transcripts_dir = config.TRANSCRIPTS_DIR
    os.makedirs(transcripts_dir, exist_ok=True)

    # Mock mode: skip ffmpeg and LLM entirely, return canned transcript
    if config.PROVIDER == "mock":
        status_callback("Mock provider: returning canned transcript...")
        time.sleep(0.5)  # simulate processing time so UI state transitions are visible
        transcript = _MOCK_TRANSCRIPT
        transcript_path = os.path.join(transcripts_dir, f"{recording_name}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        logger.info("Mock transcript saved: %s", transcript_path)
        status_callback("Pipeline complete.")
        return transcript

    # MKV always needs conversion; Azure Speech also works best with WAV
    ext = Path(audio_path).suffix.lower()
    if ext == ".mkv":
        status_callback("Converting MKV to WAV...")
        save_dir = wav_dir or config.RECORDINGS_DIR
        os.makedirs(save_dir, exist_ok=True)
        upload_path = os.path.join(save_dir, f"{recording_name}.wav")
        _convert_to_wav(audio_path, upload_path)
        logger.info("WAV ready: %s", upload_path)
        status_callback(f"Audio saved: {os.path.basename(upload_path)}")
    else:
        upload_path = audio_path

    if config.PROVIDER == "azure":
        transcript = _transcribe_with_azure(upload_path, status_callback)
    else:
        transcript = _transcribe_with_gemini(upload_path, model_key, status_callback)

    transcript_path = os.path.join(transcripts_dir, f"{recording_name}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    logger.info("Transcript saved: %s", transcript_path)

    status_callback("Pipeline complete.")
    return transcript


async def run(audio_path: str, recording_name: str, status_callback: Callable[[str], None]) -> str:
    """Run the cloud pipeline asynchronously."""
    loop = asyncio.get_running_loop()

    def sync_callback(message: str) -> None:
        loop.call_soon_threadsafe(status_callback, message)

    return await loop.run_in_executor(
        None,
        lambda: _run_pipeline_sync(audio_path, recording_name, sync_callback),
    )


def _markdown_to_confluence_storage(md: str) -> str:
    """Convert basic AI-generated markdown to Confluence storage format (XHTML subset)."""
    import html as _html
    import re

    lines = md.splitlines()
    out = []
    in_list = False

    def inline(text: str) -> str:
        text = _html.escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        return text

    for line in lines:
        if line.startswith('## '):
            if in_list: out.append('</ul>'); in_list = False
            out.append(f'<h2>{inline(line[3:])}</h2>')
        elif line.startswith('### '):
            if in_list: out.append('</ul>'); in_list = False
            out.append(f'<h3>{inline(line[4:])}</h3>')
        elif re.match(r'^[-*] ', line):
            if not in_list: out.append('<ul>'); in_list = True
            out.append(f'<li>{inline(line[2:])}</li>')
        elif line.strip() == '':
            if in_list: out.append('</ul>'); in_list = False
        else:
            if in_list: out.append('</ul>'); in_list = False
            out.append(f'<p>{inline(line)}</p>')

    if in_list:
        out.append('</ul>')
    return '\n'.join(out)


def export_to_confluence(title: str, transcript: str, summary: str = "") -> str:
    """Export transcript (and optional summary) to a Confluence page.

    Creates the page if it doesn't exist under CONFLUENCE_PARENT_PAGE_ID,
    or updates it if a page with the same title already exists there.

    Returns: URL string of the created/updated page.
    Raises: RuntimeError if export fails.
    """
    import base64
    import requests as _requests

    if not config.CONFLUENCE_URL:
        raise RuntimeError("CONFLUENCE_URL is not configured")
    if not config.CONFLUENCE_EMAIL or not config.CONFLUENCE_TOKEN:
        raise RuntimeError("CONFLUENCE_EMAIL and CONFLUENCE_TOKEN are required")
    if not config.CONFLUENCE_SPACE_KEY:
        raise RuntimeError("CONFLUENCE_SPACE_KEY is not configured")

    base_url = config.CONFLUENCE_URL.rstrip('/')
    credentials = base64.b64encode(
        f"{config.CONFLUENCE_EMAIL}:{config.CONFLUENCE_TOKEN}".encode()
    ).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Build page content in Confluence storage format
    parts = []
    if summary.strip():
        parts.append('<h2>Summary</h2>')
        parts.append(_markdown_to_confluence_storage(summary))
    if config.EXPORT_INCLUDE_TRANSCRIPT and transcript.strip():
        parts.append('<h2>Full Transcript</h2>')
        parts.append(
            '<ac:structured-macro ac:name="code">'
            '<ac:parameter ac:name="language">text</ac:parameter>'
            '<ac:plain-text-body><![CDATA[' + transcript + ']]></ac:plain-text-body>'
            '</ac:structured-macro>'
        )
    if not parts:
        parts.append('<p><em>No content to export.</em></p>')
    body_storage = '\n'.join(parts)

    # Check if a page with this title already exists under the parent
    existing_id = None
    if config.CONFLUENCE_PARENT_PAGE_ID:
        search_url = f"{base_url}/wiki/rest/api/content"
        r = _requests.get(search_url, headers=headers, params={
            "title": title,
            "spaceKey": config.CONFLUENCE_SPACE_KEY,
            "ancestor": config.CONFLUENCE_PARENT_PAGE_ID,
            "expand": "version",
        })
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                existing_id = results[0]["id"]
                existing_version = results[0]["version"]["number"]

    if existing_id:
        # Update existing page
        payload = {
            "version": {"number": existing_version + 1},
            "title": title,
            "type": "page",
            "body": {"storage": {"value": body_storage, "representation": "storage"}},
        }
        r = _requests.put(
            f"{base_url}/wiki/rest/api/content/{existing_id}",
            headers=headers, json=payload,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Confluence update failed {r.status_code}: {r.text}")
        page_id = existing_id
    else:
        # Create new page
        payload: dict = {
            "type": "page",
            "title": title,
            "space": {"key": config.CONFLUENCE_SPACE_KEY},
            "body": {"storage": {"value": body_storage, "representation": "storage"}},
        }
        if config.CONFLUENCE_PARENT_PAGE_ID:
            payload["ancestors"] = [{"id": config.CONFLUENCE_PARENT_PAGE_ID}]
        r = _requests.post(
            f"{base_url}/wiki/rest/api/content",
            headers=headers, json=payload,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Confluence create failed {r.status_code}: {r.text}")
        page_id = r.json()["id"]

    return f"{base_url}/wiki/spaces/{config.CONFLUENCE_SPACE_KEY}/pages/{page_id}"


def export_to_notion(title: str, transcript: str, summary: str = "") -> str:
    """
    Export transcript to Notion.

    MOCK IMPLEMENTATION — replace with real Notion API calls when ready.

    Real implementation should:
    1. Use NOTION_TOKEN, NOTION_DATABASE_ID from config
    2. POST to https://api.notion.com/v1/pages
    3. Set parent.database_id = NOTION_DATABASE_ID
    4. Add title property, transcript and summary as paragraph blocks
    5. Return the URL of the created page

    Returns: URL string of the created page.
    Raises: RuntimeError if export fails.
    """
    if not config.NOTION_TOKEN:
        raise RuntimeError("NOTION_TOKEN is not configured")
    logger.info("MOCK: Would export '%s' to Notion database %s", title, config.NOTION_DATABASE_ID)
    return f"https://notion.so/mock-page-{title.replace(' ', '-')}"
