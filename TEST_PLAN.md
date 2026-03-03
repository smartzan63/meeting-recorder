# Meeting Recorder — Test Plan

AI-assisted testing strategy. Tests are executed by Claude in Chrome using the scenarios in `SCENARIOS.md` as the spec. No Playwright scripts to maintain.

---

## Philosophy

Traditional test pyramid inverted for AI-assisted development:

- **Most tests** live at the E2E UI layer — Claude drives the browser, asserts on real rendered output
- **Fewest tests** at the unit layer — not worth the maintenance cost when AI can verify behavior through the UI
- Tests prove the system works to a human level of confidence, not a code-coverage metric

---

## Two Levels

### Level 1 — Semi-mocked E2E (bulk of tests)

**What runs:** Real Docker container, real FastAPI backend, real React frontend.

**What is mocked:**
- OBS — replaced by a fake WebSocket server on port `4456`. Responds to `StartRecord`/`StopRecord`, copies a fixture file into `data/audio/` on stop, returns a valid file path. Real OBS stays on `4455` and can run simultaneously without conflict.
- LLM provider — `PROVIDER=mock` mode in the backend returns a canned transcript instantly with no API call. Transcript includes two speakers (`SPEAKER_00`, `SPEAKER_01`) so the speaker editor is exercised in every test run.

**What is tested:**
All scenarios in `SCENARIOS.md` except the real OBS happy path (S2.1–S2.4 with real audio), which is covered by Level 2.

**When it runs:** On demand — run before any UI change is considered done.

**How to run Level 1:**
1. `docker-compose -f docker-compose.test.yml up --build`
2. Open http://localhost:8181
3. Run scenarios from SCENARIOS.md using Claude in Chrome (or manually)
4. App uses mock OBS (port 4456) and PROVIDER=mock — no real OBS or LLM keys needed

**Infrastructure:**
- `test/mock_obs_server.py` — fake OBS WebSocket server (obsws v5 protocol)
- `config.py` / `pipeline.py` PROVIDER=mock — returns canned transcript instantly
- `docker-compose.test.yml` — wires everything together

---

### Level 2 — Real smoke test (1 test, pre-push gate)

**What runs:** Everything real — real OBS, real LLM provider (Azure or Gemini per `.env`), real Docker container.

**What is tested:** The main recording flow end-to-end:
1. Claude clicks Start Recording
2. System plays `test/fixtures/smoke.wav` (~10s, "Testing recording, one two three") through speakers
3. OBS captures it via WASAPI Desktop Audio
4. Claude clicks Stop Recording, names the recording, clicks Process
5. Claude asserts the transcript is non-empty and contains expected words ("testing", "recording")

**Audio fixture:** `test/fixtures/smoke.wav` — committed to the repo. A ~10 second recording of "Testing recording, one, two, three." Fully automated, no human speaking required.

**When it runs:** Once, immediately before `git push`. OBS must be running with desktop audio configured.

---

## Infrastructure Needed

| Component | Purpose | Port |
|---|---|---|
| App container | Real FastAPI + React | 8080 |
| Mock OBS server | Fake WebSocket, responds to record commands | 4456 |

App started with `OBS_PORT=4456` and `PROVIDER=mock` for Level 1 runs.

---

## Mock Provider — canned transcript

When `PROVIDER=mock`, the backend returns this fixed transcript immediately (no API call):

```
[00:00] SPEAKER_00
This is a test recording.

[00:05] SPEAKER_01
Hello from speaker two. The quick brown fox.

[00:10] SPEAKER_00
Testing complete.
```

Used to verify transcript display, speaker editor, copy, summarize, and history flows.

---

## Open Questions (Resolved)

1. **Level 1 run trigger** — on demand only. The docker-compose approach (`docker-compose -f docker-compose.test.yml up --build`) makes it explicit and intentional.

2. **Test environment isolation** — separate container with isolated named volumes (`data/audio`, `data/transcripts`, `data/summaries`). Test transcripts never appear in the real history. Resolved by `docker-compose.test.yml`.

3. **Mock transcript format** — two speakers (`SPEAKER_00`, `SPEAKER_01`) as shown in the canned transcript above, so the speaker editor is exercised on every Level 1 run.

4. **`smoke.wav` source** — recorded separately, not committed. See `test/fixtures/README.md` for instructions on where to place it.

5. **Level 2 assertion strength** — "transcript contains specific words": check that the transcript contains "testing" and "recording". More meaningful than non-empty, and robust enough across providers given the controlled audio.

---

## Known Bugs (to be covered by tests once fixed)

- **Timer desync (open):** App timer resets to 0:00 whenever the client enters `recording` state — see `BUG_TIMER_DESYNC.md` for root cause and fix plan. Scenario S2.1 should assert timer matches actual OBS elapsed time on reconnect.
