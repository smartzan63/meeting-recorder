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
- OBS — replaced by a fake WebSocket server on port `4456`. Responds to `StartRecord`/`StopRecord`, copies a fixture file into `recordings/` on stop, returns a valid file path. Real OBS stays on `4455` and can run simultaneously without conflict.
- LLM provider — `PROVIDER=mock` mode in the backend returns a canned transcript instantly with no API call. Transcript includes two speakers (`SPEAKER_00`, `SPEAKER_01`) so the speaker editor is exercised in every test run.

**What is tested:**
All scenarios in `SCENARIOS.md` except the real OBS happy path (S2.1–S2.4 with real audio), which is covered by Level 2.

**When it runs:** On demand — run before any UI change is considered done.

**How to run:** TBD (docker compose profile or separate env file — see Open Questions).

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

## Open Questions

These need to be decided before implementation starts:

1. **Level 1 run trigger** — on demand only, or also automatically before commit (like Level 2)?

2. **Test environment isolation** — should Level 1 tests run against a separate container with isolated volumes (so test transcripts don't appear in real history), or is the same container acceptable if tests clean up after themselves?

3. **`smoke.wav` source** — use a committed placeholder that gets replaced, or record one now and commit it?

4. **Level 2 smoke test assertion strength** — "transcript is non-empty" or "transcript contains specific words"? Specific words is more meaningful but fragile across providers.

---

## Known Bugs (to be covered by tests once fixed)

- **Timer desync (open):** App timer resets to 0:00 whenever the client enters `recording` state — see `BUG_TIMER_DESYNC.md` for root cause and fix plan. Scenario S2.1 should assert timer matches actual OBS elapsed time on reconnect.
