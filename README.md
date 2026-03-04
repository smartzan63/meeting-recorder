# meeting-recorder

Web UI for recording meetings from any platform (Zoom, Slack, Teams, Meet…) and getting a speaker-labeled transcript automatically. OBS captures microphone and system audio; a cloud AI service transcribes and diarizes in one pass.

Open `http://localhost:8080`, click **Start Recording**, talk, click **Stop Recording** — a speaker-labeled transcript appears in the browser within seconds of upload completing.

## Features

- One-button recording control via OBS WebSocket
- Cloud transcription + speaker diarization — no local GPU needed
- Speaker name editor — rename `SPEAKER_00` / `SPEAKER_01` to real names; names are saved automatically and restored on next load
- AI speaker enrichment — identifies real names from conversation context and pre-fills the name editor; user edits are always preserved through re-enrichment
- Enrich & Summarise — generates a structured English summary regardless of transcript language; auto-saved to disk; shows a warning when speaker names have changed since the last summary
- File upload — process any audio/video file (M4A, WAV, MP4, MKV…) without OBS
- Transcript history — collapsible panel of past runs; click to reload transcript, speaker names, and summary
- Export to Confluence (real REST API) or Notion — speaker substitution is applied server-side at export time, so correcting a name instantly reflects in all future exports without re-running enrichment; optional full transcript via `EXPORT_INCLUDE_TRANSCRIPT`
- Three AI provider options: **Google Gemini**, **Azure AI Speech + Azure OpenAI**, and **Mock** (no API keys, instant canned output for UI testing)
- Runs in Docker — no Python environment setup required

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [OBS Studio](https://obsproject.com/) installed and running on the host
- An API key for your chosen provider (see [Providers](#providers) below)

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/smartzan63/meeting-recorder.git
cd meeting-recorder

# 2. Create .env from the template
cp .env.example .env
# Edit .env — set PROVIDER and fill in credentials for your chosen provider

# 3. Start the container
docker compose up -d

# 4. Open the UI
open http://localhost:8080   # macOS
start http://localhost:8080  # Windows
```

OBS must be running on the host machine before you click **Start Recording**.

### Manage the container

```bash
docker compose up -d --build   # rebuild after code changes
docker compose up -d --force-recreate  # pick up .env changes without rebuilding
docker compose logs -f         # tail logs
docker compose down            # stop
```

Recordings, transcripts, and summaries are stored under `./data/` (volume-mounted) and persist across container restarts.

## OBS setup (one-time)

### macOS

1. Install [BlackHole 2ch](https://existential.audio/blackhole/) and create a Multi-Output Device in **Audio MIDI Setup** combining your speakers and BlackHole. Set this as your system output so system audio routes through BlackHole into OBS.
2. In OBS add two audio sources: **Audio Input Capture** (microphone) and **Audio Input Capture** (BlackHole 2ch).

### Windows

1. No extra audio driver needed — OBS captures system audio natively via WASAPI.
2. In OBS add two audio sources: **Audio Input Capture** (microphone) and **Audio Output Capture** (Desktop Audio / WASAPI).

### Both platforms

3. Set the OBS recording output path to `<project root>/data/audio/` — this folder is volume-mounted into the container.
4. Enable the WebSocket server: **Tools → WebSocket Server Settings → Enable WebSocket Server**. Set port `4455` and a password of your choice.

## Providers

Set `PROVIDER=gemini`, `PROVIDER=azure`, or `PROVIDER=mock` in your `.env`.

### Gemini (default)

Requires a [Gemini API key](https://aistudio.google.com/apikey). Free tier available; connect a billing account for higher quotas and better availability.

Transcription, diarization, speaker name enrichment, and summarization all use Gemini.

Available models (set via the UI dropdown at runtime — all operations use the selected model):
- `gemini-3-flash-preview` — default; low free-tier quota and subject to demand spikes
- `gemini-2.5-flash` — recommended; higher quota and more stable availability

### Azure

Requires two Azure resources:

| Resource | Purpose |
|---|---|
| Azure AI Speech (S0) | Transcription + speaker diarization via Fast Transcription API |
| Azure OpenAI | Summarization and speaker name enrichment — deploy a chat-completions model (e.g. `gpt-5.2`) |

Azure AI Speech Fast Transcription API is recommended by Microsoft for meeting recordings — it handles files up to 1 GB with no file size limit, unlike the 25 MB cap on Azure OpenAI audio endpoints.

## Configuration

Copy `.env.example` to `.env`. The file is never committed (`.gitignore`).

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `gemini` | AI provider: `gemini`, `azure`, or `mock` |
| `OBS_PASSWORD` | — | OBS WebSocket password |
| `OBS_HOST` | `localhost` | OBS WebSocket host |
| `OBS_PORT` | `4455` | OBS WebSocket port |
| `GEMINI_API_KEY` | — | Required when `PROVIDER=gemini` |
| `AZURE_SPEECH_KEY` | — | Required when `PROVIDER=azure` |
| `AZURE_SPEECH_REGION` | — | Azure Speech resource region (e.g. `westeurope`) |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.2` | Azure OpenAI deployment name |
| `RECORDINGS_DIR` | `./data/audio` | Where audio files are saved |
| `TRANSCRIPTS_DIR` | `./data/transcripts` | Where transcript `.txt` and `.json` files are saved |
| `SUMMARIES_DIR` | `./data/summaries` | Where summary files are auto-saved |
| `CONFLUENCE_URL` | — | Confluence base URL (e.g. `https://yourcompany.atlassian.net`) |
| `CONFLUENCE_EMAIL` | — | Atlassian account email |
| `CONFLUENCE_TOKEN` | — | Atlassian API token |
| `CONFLUENCE_SPACE_KEY` | — | Confluence space key (e.g. `ENG`) |
| `CONFLUENCE_PARENT_PAGE_ID` | — | ID of the parent page for exported meeting notes |
| `NOTION_TOKEN` | — | Notion internal integration token (create at [notion.so/my-integrations](https://www.notion.so/my-integrations)) |
| `NOTION_DATABASE_ID` | — | ID of the Notion database to add pages to — visible in the database URL after the last `/` and before `?v=` |
| `EXPORT_INCLUDE_TRANSCRIPT` | `false` | Set to `true` to include full transcript in Confluence/Notion exports |
| `PORT` | `8080` | Web server port |

### Notion setup

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → **New integration** → copy the token into `NOTION_TOKEN`
2. Open your target database in Notion → `···` menu → **Connections** → add your integration
3. Copy the database ID from the URL into `NOTION_DATABASE_ID`

Both variables must be set for the Notion export button to appear in the UI.

## Architecture

```
OBS (host) ──websocket──▶ app.py (FastAPI + WebSocket)
                                │
                          pipeline.py
                                │
                    ffmpeg (MKV/MP4 → WAV)
                                │
            ┌───────────────────┼───────────────────┐
       PROVIDER=gemini    PROVIDER=azure        PROVIDER=mock
            │                   │                    │
   Gemini Files API    Azure AI Speech          canned output
   generate_content    Fast Transcription API   (no API call)
   (transcription +    (transcription +
    diarization +       diarization)
    enrichment +             │
    summarization)   Azure OpenAI
                     (enrichment +
                      summarization)
            └───────────────────┼───────────────────┘
                                │
              data/transcripts/{name}.txt + .json
              data/summaries/{name}.txt
                                │
                    WebSocket ──▶ browser UI (React)
```

| Component | Tool |
|---|---|
| Audio capture | OBS Studio |
| macOS system audio | BlackHole 2ch |
| Windows system audio | WASAPI Desktop Audio (built into OBS) |
| OBS control | `obsws-python` (WebSocket port 4455) |
| Backend | Python FastAPI + WebSocket |
| Frontend | React + Vite + Tailwind (served as static build) |
| Audio conversion | ffmpeg |

## Scenarios and test plan

`SCENARIOS.md` in the root of the repo describes all user-facing features and expected behavior. It is the source of truth for manual testing and future automated test coverage.

## Testing without OBS

Use the **Choose File** / **Process File** section in the UI to upload any audio/video file directly — no OBS needed.

For UI testing with no API keys at all, set `PROVIDER=mock` in `.env` — the backend returns a canned two-speaker transcript instantly.

Or run the pipeline directly from the command line:

```bash
python test_pipeline.py path/to/recording.wav
```

## Linux Docker note

`host.docker.internal:host-gateway` is already in `docker-compose.yml` so Linux is supported out of the box — no extra steps needed.

