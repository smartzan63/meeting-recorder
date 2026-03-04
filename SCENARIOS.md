# Meeting Recorder — Feature Scenarios

Describes all user-facing features and the expected behavior for each scenario.
Intended as the source of truth for manual testing and future Playwright test coverage.

App runs at `http://localhost:8080`. Backend: FastAPI + WebSocket. Provider set via `PROVIDER` env var (`gemini`, `azure`, or `mock`).

---

## 1. App Load

**S1.1 — Initial state**
- Page loads, header shows "Meeting Recorder" and a provider badge ("Gemini", "Azure", or "Mock")
- Left panel shows the record button (green, "Start Recording")
- Right panel shows an empty state: icon + "Start a recording or upload a file"
- Model selector is visible if more than one model is available (Gemini provider only)
- History section is hidden if no past transcripts exist

**S1.2 — History loaded on boot**
- If past transcripts exist, the History section appears in the right panel (collapsed by default)
- History count in the trigger label matches the number of stored transcripts

---

## 2. OBS Recording Flow

Prerequisites: OBS is running, WebSocket enabled on port 4455, recording output path set to `<repo>/data/audio/`.

**S2.1 — Start recording**
- Click the green "Start Recording" button
- Button turns red with pulsing ring, label changes to "Stop Recording"
- Timer starts counting up from 00:00
- Status message shows "Recording…"
- Model selector becomes disabled

**S2.2 — Stop recording**
- Click "Stop Recording" while recording
- Button immediately changes to grey "Stopping…" (disabled), timer hides
- After backend responds (~5–20s while OBS finishes writing the file), a "Save Recording" dialog appears
- Dialog is pre-filled with a default filename derived from the recording timestamp
- Dialog cannot be dismissed by clicking outside

**S2.3 — Save with custom name**
- User edits the filename in the dialog
- Clicks "Save" (or presses Enter)
- Dialog closes, a "Process now?" prompt appears with the saved path
- "Process" and "Skip" buttons are shown

**S2.4 — Cancel save dialog**
- Clicking "Cancel" or the X button on the Save Recording dialog closes it
- App returns to idle state — no `.wav` file is created at this point
- Record button becomes available again

**S2.5 — Process immediately**
- Click "Process"
- Status message shows "Transcribing…" with a spinner
- Record button is disabled
- When pipeline completes, transcript appears in the right panel
- History list updates with the new entry

**S2.6 — Skip processing**
- Click "Skip" on the process prompt
- App returns to idle state, no transcript is shown
- The saved `.wav` file remains in `data/audio/`

**S2.7 — Stop recording fails (OBS not recording)**
- If OBS returns an error on stop (e.g. was not actually recording)
- Button resets to green "Start Recording"
- Error message shown below the button in red
- App is fully usable — user can start a new recording

---

## 3. File Upload Flow

**S3.1 — Choose and process a file**
- Click "Choose File", select an audio/video file (M4A, WAV, MP4, MKV)
- Filename appears on the button label
- Click "Process File"
- Status shows "Uploading…" then "Transcribing…" with spinner
- Record button and file controls become disabled during processing
- Transcript appears when done

**S3.2 — File upload while processing is active**
- If pipeline is already running, "Process File" button remains disabled
- Attempting to upload via API returns 409 Conflict

**S3.3 — Unsupported or corrupt file**
- Pipeline fails and an error message is shown
- App returns to usable state after Dismiss

---

## 4. Error State Recovery

**S4.1 — Error state shown**
- When the pipeline fails (e.g. Gemini 503, OBS timeout), the status line shows "An error occurred." in red
- A "Dismiss" link appears inline next to the error message

**S4.2 — Dismiss error**
- Clicking "Dismiss" calls `POST /reset` to reset server state to idle
- Status message clears, record button becomes available again
- App is fully usable after dismissal

---

## 5. Transcript Display

**S5.1 — Transcript renders after pipeline**
- Right panel shows the transcript text in a scrollable area
- Header row shows "Transcript" label, model badge, and a Copy button
- Speaker editor appears below the header if `SPEAKER_XX` labels are present

**S5.2 — Speaker name editor**
- One input row per unique `SPEAKER_XX` detected at transcription time (stored in meta, not rescanned from text)
- Each speaker has a distinct color badge
- Typing a name into an input immediately replaces all occurrences of that speaker label in the displayed transcript
- Original transcript is preserved with `SPEAKER_XX` labels in storage — clearing the input restores the label in display
- Copy button copies the transcript with speaker names applied (not the raw `SPEAKER_XX` text)

**S5.3 — Speaker names persist across reload**
- Speaker names are saved to the backend on every keystroke via `PUT /transcripts/{id}` with `{"speakers": {...}}`
- Names are stored in the transcript meta JSON (`data/transcripts/{name}.json`)
- Reloading the page and loading the same recording from history restores the saved names

**S5.4 — Copy transcript**
- Click "Copy"
- Button briefly shows "Copied" then reverts
- Clipboard contains the current displayed transcript text (with any speaker name substitutions)

---

## 6. Summarization

**S6.1 — Generate summary**
- With a transcript loaded, click "Enrich & Summarise"
- Button shows "Enriching…" with a spinner and disables during the request
- Summary card appears below the transcript when the response arrives
- Summary is rendered as formatted markdown
- Summary is always generated in English regardless of the transcript language
- Summary is auto-saved to `data/summaries/{name}.txt` if the recording has a name

**S6.2 — Toggle raw / formatted**
- Summary card has a "Raw" button
- Clicking it switches to plain text display of the markdown source
- Clicking "Formatted" switches back to rendered markdown

**S6.3 — Dismiss summary**
- Click the X button on the summary card
- Card disappears, transcript remains visible

**S6.4 — Re-summarize**
- After dismissing, the "Enrich & Summarise" button is available again
- When a summary already exists, the button reads "Re-enrich & Summarise"
- Clicking it generates a new summary and overwrites the saved file

**S6.5 — Summary outdated banner**
- If the user edits a speaker name after a summary has been generated, an amber warning appears above the summary card: "Speaker names changed — summary may be outdated. Re-enrich & Summarise to update."
- The warning disappears when a new summary is generated via "Re-enrich & Summarise"
- The warning disappears when a different recording is loaded from history

---

## 7. History

**S7.1 — History panel collapsed by default**
- History section is shown but collapsed on load
- Clicking the "History (N)" trigger expands it

**S7.2 — Load a past transcript**
- Click anywhere on a history tile
- Transcript from that session loads in the right panel
- Speaker editor rebuilds with the speakers stored at transcription time
- Previously saved speaker names are restored from meta
- If a summary was saved for that recording, the summary card also loads automatically

**S7.3 — Delete a recording**
- Hover over a history tile — trash icon appears on the right
- Click the trash icon (does not load the transcript — click is isolated)
- Confirmation dialog appears: "Remove recording?" with the recording name
- Clicking "Cancel" closes the dialog, recording remains
- Clicking "Remove" deletes all associated files: `data/transcripts/{name}.txt`, `data/transcripts/{name}.json`, `data/summaries/{name}.txt` (if exists), `data/audio/{name}.wav` (if exists)
- History list refreshes; if the deleted item was the last one, the History section disappears

**S7.4 — Summary indicator on history tiles**
- History tiles show a small "Summary" badge when `has_summary: true` is returned by the backend
- Badge indicates a saved summary is available and will be loaded when the tile is clicked

---

## 8. Model Selection (Gemini provider only)

**S8.1 — Model selector shown for multiple models**
- When `PROVIDER=gemini`, the selector shows available Gemini models
- Default model is pre-selected

**S8.2 — Model selector hidden for single-model provider**
- When `PROVIDER=azure`, only "Azure AI Speech" is available and the selector is hidden

**S8.3 — Model selector disabled while busy**
- Selector is disabled during recording, stopping, and transcribing states

---

## 9. WebSocket State Sync

**S9.1 — Reconnect restores state**
- If the browser loses and regains the WebSocket connection, the UI syncs to the server's current state
- If server is idle → UI goes to idle
- If server is transcribing → UI shows transcribing with spinner
- If server reports recording but client is in an unexpected state → client stays in its current state

**S9.2 — Live status messages during transcription**
- While the pipeline runs, incremental status messages (e.g. "Converting audio…", "Uploading to API…") appear in the status line

---

## 10. Provider Badge

**S10.1 — Badge reflects active provider**
- Header badge shows "Gemini" when `PROVIDER=gemini`
- Header badge shows "Azure" when `PROVIDER=azure`
- Badge is informational only — provider cannot be changed from the UI

---

## 11. AI Speaker Enrichment

**S11.1 — Enrich runs before summarization**
- With a transcript loaded, clicking "Enrich & Summarise" first calls `/enrich`, then `/summarize`
- While the operation is in progress the button shows "Enriching…" with a spinner and is disabled

**S11.2 — Names identified and populated**
- If the transcript contains speaker names in context (e.g. someone introduces themselves or is addressed by name), the speaker name inputs are auto-populated with the identified names
- The displayed transcript immediately substitutes the identified names in place of `SPEAKER_XX` labels
- Enriched names are only applied to inputs that are currently empty — names already typed by the user are not overwritten

**S11.3 — Summary uses identified names**
- Summary is generated after enrichment completes
- The enriched transcript (with real names) is passed to `/summarize`, so the summary reflects real names
- Summary is always returned in English

**S11.4 — Graceful fallback**
- If enrichment returns no names (names not found in transcript, or Gemini unavailable), summarization still completes normally
- Speaker name inputs remain empty; no error is shown
- The `/enrich` endpoint always returns HTTP 200 — never 500

---

## 12. Transcript Export

**S12.1 — Download**
- With a transcript loaded, click "Download .txt"
- Browser downloads a file named `transcript.txt`
- File contents match the current transcript with speaker name substitutions applied (not raw `SPEAKER_XX` text)
- No backend call is made — download is entirely client-side

**S12.2 — Confluence export**
- Click "Confluence" with a transcript loaded
- Button is only shown when `CONFLUENCE_URL`, `CONFLUENCE_TOKEN`, etc. are configured
- Backend loads the original `SPEAKER_XX` transcript from storage, applies the latest saved speaker mapping, and exports the substituted text
- Backend creates a Confluence page via REST API and returns the page URL
- URL is displayed below the export buttons as a link: `Exported: <url>`
- Export content includes the summary (if available); full transcript is included only when `EXPORT_INCLUDE_TRANSCRIPT=true`

**S12.3 — Notion export**
- Click "Notion" with a transcript loaded
- Button is only shown when `NOTION_TOKEN` and `NOTION_DATABASE_ID` are configured
- Backend loads the original `SPEAKER_XX` transcript from storage, applies the latest saved speaker mapping, and exports the substituted text
- Backend creates a Notion page in the configured database and returns the page URL
- Page title is derived from the source filename with timestamp prefix stripped
- Summary and transcript are exported with full markdown formatting (headings, bullets, bold)
- URL is displayed below the export buttons as a link: `Exported: <url>`

**S12.4 — Export uses server-side substitution**
- Exports always load the original `SPEAKER_XX` transcript from storage and apply the current saved speaker mapping server-side
- This means: correcting a speaker name in the UI and clicking Export immediately uses the corrected name — no re-enrichment needed
- Raw `SPEAKER_XX` labels are not present in exported content when names have been assigned

**S12.5 — Export disabled during processing**
- All export buttons (Download .txt, Confluence, Notion) are disabled when transcription is in progress
- Confluence and Notion buttons show a spinner while `isExporting` is true

**S12.6 — Export error**
- If `/export` returns an error (e.g. credentials not configured), the error message is shown in the status area
- Export buttons return to their normal enabled state after Dismiss

---

## Known Limitations

- One recording at a time — starting a new recording while one is transcribing returns 409
- No authentication — app is intended for local/trusted-network use
- OBS must be running before clicking Start Recording; if not connected, a 503 error is shown
- OBS recording output path must be set to `<repo>/data/audio/` — the default OBS path will not work with the Docker volume mount
- Gemini free tier has a request quota; 503 errors will appear if the quota is exceeded
- Runtime data stored under `data/audio/`, `data/transcripts/`, `data/summaries/` — not committed to git
