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


_ENRICH_PROMPT = (
    "You are analyzing a meeting transcript. Identify the real names of the speakers "
    "if they can be determined from the conversation context (e.g. someone is addressed "
    "by name, introduces themselves, or signs off with their name).\n\n"
    "Return ONLY a JSON object mapping speaker labels to names, e.g.:\n"
    '{{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}}\n\n'
    "If a speaker's name cannot be determined, omit them from the result.\n"
    "If no names can be determined at all, return an empty object: {{}}\n\n"
    "Do not include any explanation, only the JSON object.\n\n"
    "Transcript:\n{transcript}"
)


def _parse_speaker_json(raw: str) -> dict[str, str]:
    import json, re
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if isinstance(k, str) and isinstance(v, str)}
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: extract first {...} block from the response
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(k, str) and isinstance(v, str)}
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def enrich_transcript(transcript: str, model_key: str = config.DEFAULT_MODEL) -> dict[str, str]:
    """Identify real speaker names from transcript context. Returns a mapping like
    {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"} or {} if names cannot be determined.

    Uses Azure OpenAI when PROVIDER=azure, Gemini otherwise.
    Never raises — returns {} on any error.
    """
    if config.PROVIDER == "mock":
        return {}

    try:
        if config.PROVIDER == "azure":
            if not config.AZURE_OPENAI_KEY:
                logger.warning("enrich_transcript: AZURE_OPENAI_KEY not set, skipping enrichment")
                return {}

            from openai import AzureOpenAI

            client = AzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_KEY,
                api_version="2025-01-01-preview",
            )
            prompt = _ENRICH_PROMPT.format(transcript=transcript)
            response = client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()
            return _parse_speaker_json(raw)

        else:
            # Gemini provider
            if not config.GEMINI_API_KEY:
                logger.warning("enrich_transcript: GEMINI_API_KEY not set, skipping enrichment")
                return {}

            from google import genai  # type: ignore

            model_id = config.MODELS.get(model_key, config.MODELS[config.DEFAULT_MODEL])["model"]
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            prompt = _ENRICH_PROMPT.format(transcript=transcript)
            # Do NOT use response_mime_type="application/json" — it causes the SDK
            # to misbehave and raise exceptions on valid responses.
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt],
            )
            raw = response.text
            logger.info("enrich_transcript raw response: %r", raw)
            result = _parse_speaker_json(raw)
            logger.info("enrich_transcript parsed result: %r", result)
            return result

    except Exception as e:
        logger.warning("enrich_transcript failed (non-fatal): %s", e, exc_info=True)
        return {}


_MOCK_SUMMARY = """\
## Summary

**Key topics:** Test recording, speaker identification

**Decisions made:** None

**Action items:** None"""


def summarize_transcript(transcript: str, model_key: str = config.DEFAULT_MODEL) -> str:
    """Summarize a transcript using the configured provider. Returns markdown."""
    if config.PROVIDER == "mock":
        return _MOCK_SUMMARY
    if config.PROVIDER == "azure":
        return _summarize_with_azure(transcript)
    return _summarize_with_gemini(transcript, model_key)


def _summarize_with_gemini(transcript: str, model_key: str = config.DEFAULT_MODEL) -> str:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai  # type: ignore

    model_id = config.MODELS.get(model_key, config.MODELS[config.DEFAULT_MODEL])["model"]
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model_id,
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
    return f"""Summarize this meeting transcript. Be concise and structured. Always respond in English regardless of the language of the transcript.
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


def _rich_text(text: str) -> list[dict]:
    """Parse **bold** markers into a Notion rich_text array."""
    import re
    parts: list[dict] = []
    for seg in re.split(r'(\*\*[^*]+\*\*)', text):
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**') and len(seg) > 4:
            parts.append({"type": "text", "text": {"content": seg[2:-2]}, "annotations": {"bold": True}})
        else:
            parts.append({"type": "text", "text": {"content": seg}})
    return parts or [{"type": "text", "text": {"content": text}}]


def _notion_heading(level: int, text: str) -> dict:
    htype = f"heading_{level}"
    return {"object": "block", "type": htype, htype: {"rich_text": _rich_text(text.strip())}}


def _notion_bullet(text: str, children: list | None = None) -> dict:
    payload: dict = {"rich_text": _rich_text(text.strip())}
    if children:
        payload["children"] = children
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": payload}


def _notion_paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text(text.strip())}}


def _markdown_to_notion_blocks(markdown: str) -> list[dict]:
    """Convert markdown text to Notion block objects.

    Handles: # headings, * / - bullets (2-level nesting), **bold**, plain paragraphs.
    Chunks paragraph text at 2000 chars to respect Notion's rich_text limit.
    """
    blocks: list[dict] = []
    lines = markdown.split('\n')
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        indent = len(raw) - len(raw.lstrip())
        i += 1

        if not line.strip():
            continue

        # Headings
        if line.lstrip().startswith('### '):
            blocks.append(_notion_heading(3, line.lstrip()[4:]))
        elif line.lstrip().startswith('## '):
            blocks.append(_notion_heading(2, line.lstrip()[3:]))
        elif line.lstrip().startswith('# '):
            blocks.append(_notion_heading(1, line.lstrip()[2:]))

        # Top-level bullet
        elif indent < 4 and (line.lstrip().startswith('* ') or line.lstrip().startswith('- ')):
            content = line.lstrip()[2:]
            children: list[dict] = []
            while i < len(lines):
                nxt = lines[i]
                nxt_indent = len(nxt) - len(nxt.lstrip())
                nxt_stripped = nxt.lstrip()
                if nxt_stripped and nxt_indent >= 4 and (nxt_stripped.startswith('* ') or nxt_stripped.startswith('- ')):
                    children.append(_notion_bullet(nxt_stripped[2:]))
                    i += 1
                else:
                    break
            blocks.append(_notion_bullet(content, children or None))

        # Indented bullet not caught above (orphan)
        elif line.lstrip().startswith('* ') or line.lstrip().startswith('- '):
            blocks.append(_notion_bullet(line.lstrip()[2:]))

        # Plain paragraph — chunk at 2000 chars
        else:
            text = line.strip()
            while len(text) > 2000:
                blocks.append(_notion_paragraph(text[:2000]))
                text = text[2000:]
            if text:
                blocks.append(_notion_paragraph(text))

    return blocks


def export_to_notion(title: str, transcript: str, summary: str = "") -> str:
    """Export transcript (and optional summary) to a Notion database page.

    Creates a new page in the configured NOTION_DATABASE_ID with the transcript
    and summary as paragraph blocks. Long content is chunked to respect Notion's
    2000-character-per-text-element limit. Pages with more than 100 blocks are
    created in batches using the append-children endpoint.

    Returns: URL string of the created page.
    Raises: RuntimeError if export fails.
    """
    import requests as _requests

    if not config.NOTION_TOKEN:
        raise RuntimeError("NOTION_TOKEN is not configured")
    if not config.NOTION_DATABASE_ID:
        raise RuntimeError("NOTION_DATABASE_ID is not configured")

    headers = {
        "Authorization": f"Bearer {config.NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    # Build all blocks
    all_blocks: list[dict] = []
    if summary.strip():
        all_blocks.append(_notion_heading(2, "Summary"))
        all_blocks.extend(_markdown_to_notion_blocks(summary.strip()))
    if config.EXPORT_INCLUDE_TRANSCRIPT and transcript.strip():
        all_blocks.append(_notion_heading(2, "Full Transcript"))
        chunk_size = 2000
        for i in range(0, max(len(transcript.strip()), 1), chunk_size):
            all_blocks.append(_notion_paragraph(transcript.strip()[i:i + chunk_size]))
    if not all_blocks:
        all_blocks.append(_notion_paragraph("No content to export."))

    # Create page with up to 100 initial children (Notion API limit)
    payload = {
        "parent": {"database_id": config.NOTION_DATABASE_ID},
        "properties": {
            "Name": {"title": [{"type": "text", "text": {"content": title}}]},
        },
        "children": all_blocks[:100],
    }
    r = _requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Notion create failed {r.status_code}: {r.text}")

    page_id = r.json()["id"]

    # Append remaining blocks in batches of 100
    remaining = all_blocks[100:]
    while remaining:
        batch, remaining = remaining[:100], remaining[100:]
        r = _requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            json={"children": batch},
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Notion append blocks failed {r.status_code}: {r.text}")

    # Notion page URL uses the ID without dashes
    clean_id = page_id.replace("-", "")
    return f"https://notion.so/{clean_id}"
