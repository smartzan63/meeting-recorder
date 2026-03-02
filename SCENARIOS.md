# Meeting Recorder — Feature Scenarios

Describes all user-facing features and the expected behavior for each scenario.
Intended as the source of truth for manual testing and future Playwright test coverage.

App runs at `http://localhost:8080`. Backend: FastAPI + WebSocket. Provider set via `PROVIDER` env var (`gemini` or `azure`).

---

## 1. App Load

**S1.1 — Initial state**
- Page loads, header shows "Meeting Recorder" and a provider badge ("Gemini" or "Azure")
- Left panel shows the record button (green, "Start Recording")
- Right panel shows an empty state: icon + "Start a recording or upload a file"
- Model selector is visible if more than one model is available (Gemini provider only)
- History section is hidden if no past transcripts exist

**S1.2 — History loaded on boot**
- If past transcripts exist, the History section appears in the right panel (collapsed by default)
- History count in the trigger label matches the number of stored transcripts

---

## 2. OBS Recording Flow

Prerequisites: OBS is running, WebSocket enabled on port 4455, recording output path set to `<repo>/recordings/`.

**S2.1 — Start recording**
- Click the green "Start Recording" button
- Button turns red with pulsing ring, label changes to "Stop Recording"
- Timer starts counting up from 00:00
- Status message shows "Recording…"
- Model selector and translate toggle become disabled

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

**S2.4 — Process immediately**
- Click "Process"
- Status message shows "Transcribing…" with a spinner
- Record button is disabled
- When pipeline completes, transcript appears in the right panel
- History list updates with the new entry

**S2.5 — Skip processing**
- Click "Skip" on the process prompt
- App returns to idle state, no transcript is shown
- The saved `.wav` file remains in `recordings/`

**S2.6 — Stop recording fails (OBS not recording)**
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
- App returns to usable state (record button re-enabled)

---

## 4. Transcript Display

**S4.1 — Transcript renders after pipeline**
- Right panel shows the transcript text in a scrollable area
- Header row shows "Transcript" label, model badge, and a Copy button
- Speaker editor appears below the header if `SPEAKER_XX` labels are present

**S4.2 — Speaker name editor**
- One input row per unique `SPEAKER_XX` found in the transcript (in order of appearance)
- Each speaker has a distinct color badge
- Typing a name into an input immediately replaces all occurrences of that speaker label in the displayed transcript
- Original transcript is preserved — clearing the name input restores the `SPEAKER_XX` label
- Copy button copies the transcript with speaker names applied (not the raw `SPEAKER_XX` text)

**S4.3 — Copy transcript**
- Click "Copy"
- Button briefly shows "Copied" then reverts
- Clipboard contains the current displayed transcript text (with any speaker name substitutions)

---

## 5. Summarization

**S5.1 — Generate summary**
- With a transcript loaded, click "Summarize"
- Button disables during the request
- Summary card appears below the transcript when the response arrives
- Summary is rendered as formatted markdown

**S5.2 — Toggle raw / formatted**
- Summary card has a "Raw" button
- Clicking it switches to plain text display of the markdown source
- Clicking "Formatted" switches back to rendered markdown

**S5.3 — Dismiss summary**
- Click the X button on the summary card
- Card disappears, transcript remains visible

**S5.4 — Re-summarize**
- After dismissing, the "Summarize" button is available again
- Clicking it generates a new summary

---

## 6. History

**S6.1 — History panel collapsed by default**
- History section is shown but collapsed on load
- Clicking the "History (N)" trigger expands it

**S6.2 — Load a past transcript**
- Click anywhere on a history tile
- Transcript from that session loads in the right panel
- Status message shows "Loaded: <source name>"
- Speaker editor rebuilds for the loaded transcript

**S6.3 — Delete a recording**
- Hover over a history tile — trash icon appears on the right
- Click the trash icon (does not load the transcript — click is isolated)
- Confirmation dialog appears: "Remove recording?" with the recording name
- Clicking "Cancel" closes the dialog, recording remains
- Clicking "Remove" deletes the transcript folder from disk and refreshes the history list
- If the deleted item was the last one, the History section disappears

> **Note (2026-03-02):** Delete functionality was implemented but has not yet been validated end-to-end through automated tests. Manual smoke test should confirm the transcript folder is actually removed from `transcripts/` on disk.

---

## 7. Model Selection (Gemini provider only)

**S7.1 — Model selector shown for multiple models**
- When `PROVIDER=gemini`, the selector shows available Gemini models
- Default model is pre-selected

**S7.2 — Model selector hidden for single-model provider**
- When `PROVIDER=azure`, only "Azure AI Speech" is available and the selector is hidden

**S7.3 — Model selector disabled while busy**
- Selector is disabled during recording, stopping, and transcribing states

---

## 8. Translate to English (Gemini provider only)

**S8.1 — Toggle is hidden for Azure**
- When `PROVIDER=azure`, the "Translate to English" toggle is not shown

**S8.2 — Toggle enables translation task**
- Enable the toggle before starting a recording or uploading a file
- Pipeline runs with `task=translate` instead of `task=transcribe`
- Transcript is returned in English regardless of source language

**S8.3 — Toggle disabled while busy**
- Toggle is disabled during recording, stopping, and transcribing

---

## 9. WebSocket State Sync

**S9.1 — Reconnect restores state**
- If the browser loses and regains the WebSocket connection, the UI syncs to the server's current state
- If server is idle → UI goes to idle
- If server is transcribing → UI shows transcribing with spinner
- If server reports recording but client is in error state (e.g. failed stop) → client stays in error state, does not revert to recording

**S9.2 — Live status messages during transcription**
- While the pipeline runs, incremental status messages (e.g. "Converting audio…", "Uploading to API…") appear in the status line

---

## 10. Provider Badge

**S10.1 — Badge reflects active provider**
- Header badge shows "Gemini" when `PROVIDER=gemini`
- Header badge shows "Azure" when `PROVIDER=azure`
- Badge is informational only — provider cannot be changed from the UI

---

## Known Limitations

- One recording at a time — starting a new recording while one is transcribing returns 409
- No authentication — app is intended for local/trusted-network use
- OBS must be running before clicking Start Recording; if not connected, a 503 error is shown
- File size: Gemini free tier has a 20 requests/day limit; Azure Speech has no practical file size limit
