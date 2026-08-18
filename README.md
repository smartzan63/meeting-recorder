# meeting-recorder

Web UI for recording meetings from any platform (Zoom, Slack, Teams, Meet…) and getting a speaker-labeled transcript automatically. OBS captures microphone and system audio; a cloud AI service transcribes and diarizes in one pass.

Open `http://localhost:8080`, click **Start Recording**, talk, click **Stop Recording** — a speaker-labeled transcript appears in the browser within seconds of upload completing.

## Features

- One-button recording control via OBS WebSocket
- Cloud transcription + speaker diarization — no local GPU needed
- Speaker name editor — rename `SPEAKER_00` / `SPEAKER_01` to real names; names are saved automatically and restored on next load
- AI speaker enrichment — identifies real names from conversation context and pre-fills the name editor; user edits are always preserved through re-enrichment. Each name carries a confidence: a name the transcript states outright is filled in silently, an inferred one is marked `~`, and a guess is left blank and marked `?` rather than filled with something plausible. Hover any field to see the transcript line the name was taken from
- Optional participant roster — supply the invite list with the nicknames people are actually called out loud, and enrichment matches speakers against it instead of writing down whatever it heard. Two participants sharing a nickname is detected and left for you to resolve
- Enrich & Summarise — generates a structured English summary regardless of transcript language; auto-saved to disk; shows a warning when speaker names have changed since the last summary
- File upload — process any audio/video file (M4A, WAV, MP4, MKV…) without OBS
- Language selection — auto-detect English/Russian or explicitly select either language for new and reprocessed transcripts
- Transcript history — collapsible panel of past runs; click to reload transcript, speaker names, and summary
- Audio file management — collapsible sidebar panel listing every saved wav with processed/unprocessed status, sizes, and total disk usage; click a processed recording to load its transcript; process an unprocessed recording (backend/model/language choice) or delete any recording's audio together with its OBS source (the transcript is kept, reprocessing is forfeited); orphaned OBS source files (`.mp4` without a saved wav) are listed separately for cleanup. Deleting from History removes transcript, wav, and OBS source together
- Export to Confluence (real REST API) or Notion — speaker substitution is applied server-side at export time, so correcting a name instantly reflects in all future exports without re-running enrichment; optional full transcript via `EXPORT_INCLUDE_TRANSCRIPT`
- Provider policy is configurable per environment: Azure is the default primary, an optional Azure or Gemini fallback runs only after a primary technical failure, and Mock returns a canned transcript for UI testing
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
# Edit .env — configure the primary provider and its credentials

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

Set `PRIMARY_PROVIDER=azure`, `PRIMARY_PROVIDER=gemini`, or `PRIMARY_PROVIDER=mock` in your `.env`. Azure is the default. Set `FALLBACK_PROVIDER=azure` or `FALLBACK_PROVIDER=gemini` to retry a failed primary transcription once; leave it empty to disable fallback.

### Gemini (default)

Requires a [Gemini API key](https://aistudio.google.com/apikey). Free tier available; connect a billing account for higher quotas and better availability.

Transcription, diarization, speaker name enrichment, and summarization all use Gemini.
In Auto mode, Gemini is instructed to preserve language switches within a meeting. Azure Fast Transcription identifies one primary language per recording, so use the manual language selector for a Russian-first or English-first Azure recording.

Available models (set via the UI dropdown at runtime, which shows each model's per-1M-token audio-input and output price — all operations use the selected model):
- `gemini-3-flash-preview` — default; best value for diarization quality
- `gemini-2.5-flash` — cheaper, stable availability
- `gemini-3.1-flash-lite` — cheapest
- `gemini-3.7-flash` — newest
- `gemini-3.6-flash`, `gemini-3.5-flash` — previous Flash generations
- `gemini-3.1-pro-preview`, `gemini-2.5-pro` — highest quality, higher cost

The list and its prices are hand-maintained in `config.py` (the Gemini API does not expose pricing).

Gemini 3.6 and 3.7 Flash are on a promotional rate that Google bills through 2026-12-31, after which the list price applies (double the promotional rate). Those entries carry `promo_until` plus their list prices, and `config.py` swaps the list price in automatically once the date passes — no edit needed in January.

### Speaker naming and the participant roster

Enrichment names speakers from the transcript alone by default. That works when someone is addressed by name, and degrades badly otherwise: the model writes down the name as it was pronounced, so a mishearing becomes a speaker name, and a meeting with two Alexes can end up with both called "Alex".

An optional roster fixes both. Copy `roster.example.json` to `roster.json` inside your transcripts directory (or point `ROSTER_PATH` anywhere else) and list who was invited:

```json
{
  "people": [
    { "name": "Ada Lovelace", "aliases": ["Ada"] },
    { "name": "Alan Turing", "aliases": ["Alan", "Al"] },
    { "name": "Alonzo Church", "aliases": ["Alonzo", "Al"] }
  ]
}
```

`aliases` is where the value is: put every form a person is called out loud, including nicknames and the spellings your transcription tends to produce. Names resolve to the canonical form, so the transcript reads consistently however someone was addressed.

Sharing an alias — "Al" above — is expected. Enrichment detects the collision and refuses to auto-fill either candidate, because picking one would hide a coin flip behind a filled-in field. Same for a speaker matching nobody on the list: the roster is a strong prior, never a closed set, so someone missing from it is left blank rather than forced onto the nearest name.

Everything the model returns is checked server-side before it reaches the UI. A name is auto-filled only at high or medium confidence; low-confidence answers are shown as a placeholder suggestion you can accept or ignore. `GET /roster` reports whether a roster was found and how many people it holds, so a run without one is distinguishable from a run where it silently failed to load.

The roster is data and never ships: `roster.json` is gitignored, and the default location is your transcripts directory, which already lives outside this repository. Only `roster.example.json`, with invented names, is committed. The tool runs normally with no roster at all.

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
| `PRIMARY_PROVIDER` | `azure` | Primary provider: `azure`, `gemini`, or `mock` |
| `FALLBACK_PROVIDER` | empty | Optional `azure` or `gemini` retry after a primary technical failure |
| `OBS_PASSWORD` | — | OBS WebSocket password |
| `OBS_HOST` | `localhost` | OBS WebSocket host |
| `OBS_PORT` | `4455` | OBS WebSocket port |
| `GEMINI_API_KEY` | — | Required for Gemini primary or enabled fallback |
| `AZURE_SPEECH_KEY` | — | Required when `PRIMARY_PROVIDER=azure` |
| `AZURE_SPEECH_REGION` | — | Azure Speech resource region (e.g. `westeurope`) |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.2` | Azure OpenAI deployment name |
| `RECORDINGS_DIR` | `./data/audio` | Where audio files are saved |
| `TRANSCRIPTS_HOST_DIR` | `./data/transcripts` | Host directory bind-mounted for transcripts. Point it at any host path to store transcripts outside the project. On Windows use forward slashes and no quotes, e.g. `C:/Users/you/transcripts` |
| `TRANSCRIPTS_DIR` | `/app/external-transcripts` | In-container path the app writes transcripts to (matches the bind-mount target). Leave as-is on every OS |
| `SUMMARIES_DIR` | `./data/summaries` | Where summary files are auto-saved |
| `CONFLUENCE_URL` | — | Confluence base URL (e.g. `https://yourcompany.atlassian.net`) |
| `CONFLUENCE_EMAIL` | — | Atlassian account email |
| `CONFLUENCE_TOKEN` | — | Atlassian API token |
| `CONFLUENCE_SPACE_KEY` | — | Confluence space key (e.g. `ENG`) |
| `CONFLUENCE_PARENT_PAGE_ID` | — | ID of the parent page for exported meeting notes |
| `NOTION_TOKEN` | — | Notion internal integration token (create at [notion.so/my-integrations](https://www.notion.so/my-integrations)) |
| `NOTION_DATABASE_ID` | — | ID of the Notion database to add pages to — visible in the database URL after the last `/` and before `?v=` |
| `EXPORT_INCLUDE_TRANSCRIPT` | `false` | Set to `true` to include full transcript in Confluence/Notion exports |
| `ROSTER_PATH` | `<TRANSCRIPTS_DIR>/roster.json` | Participant roster used as a prior for speaker naming. Optional — enrichment works without it. See [Speaker naming](#speaker-naming-and-the-participant-roster) |
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
       PRIMARY_PROVIDER=gemini  PRIMARY_PROVIDER=azure  PRIMARY_PROVIDER=mock
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
         transcripts: {id}/index.json + {version}.txt + .summary.txt
         (host location set by TRANSCRIPTS_HOST_DIR)
                                │
                    WebSocket ──▶ browser UI (React)
```

Transcripts are stored per recording in a versioned directory, so re-transcribing a recording with a different model adds a new version rather than overwriting. Timestamps are stored in UTC and rendered in the browser's local timezone in 24-hour format.

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

For UI testing with no API keys at all, set `PRIMARY_PROVIDER=mock` in `.env` — the backend returns a canned two-speaker transcript instantly.

Or run the pipeline directly from the command line:

```bash
python test_pipeline.py path/to/recording.wav
```

## Linux Docker note

`host.docker.internal:host-gateway` is already in `docker-compose.yml` so Linux is supported out of the box — no extra steps needed.
