import os
from dotenv import load_dotenv

load_dotenv()

# OBS websocket settings
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")

# Storage directories
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "./data/audio")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "./data/transcripts")
SUMMARIES_DIR = os.getenv("SUMMARIES_DIR", "./data/summaries")

# Provider: "gemini", "azure", or "mock" — independent of OS
# "mock" returns canned transcript/summary instantly, no API keys required
PROVIDER = os.getenv("PROVIDER", "gemini")

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Curated model list. The Gemini API does NOT expose pricing, so the per-1M
# token rates below are maintained by hand from the official pricing page
# (https://ai.google.dev/gemini-api/docs/pricing, last checked 2026-06-24).
# For this tool the cost drivers are audio input (the recording) and output
# (the transcript), so each entry carries an audio rate where Google bills one
# separately; audio_per_1m=None means audio is billed at the flat input rate.
# Pro models add a >200k-token tier not modelled here (meetings rarely exceed
# it). Gemini 3 Flash stays the default: best value for diarization quality.
# Run the /models/available freshness check to spot newer models Google ships.
MODELS = {
    "gemini": {
        "label": "Gemini 3 Flash (recommended)",
        "model": "gemini-3-flash-preview",
        "input_per_1m": 0.50,
        "audio_per_1m": 1.00,
        "output_per_1m": 3.00,
    },
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "input_per_1m": 0.30,
        "audio_per_1m": 1.00,
        "output_per_1m": 2.50,
    },
    "gemini-3.1-flash-lite": {
        "label": "Gemini 3.1 Flash-Lite (cheapest)",
        "model": "gemini-3.1-flash-lite",
        "input_per_1m": 0.25,
        "audio_per_1m": 0.50,
        "output_per_1m": 1.50,
    },
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash (newest, pricier)",
        "model": "gemini-3.5-flash",
        "input_per_1m": 1.50,
        "audio_per_1m": 1.50,
        "output_per_1m": 9.00,
    },
    "gemini-3.1-pro": {
        "label": "Gemini 3.1 Pro (highest quality)",
        "model": "gemini-3.1-pro-preview",
        "input_per_1m": 2.00,
        "audio_per_1m": None,
        "output_per_1m": 12.00,
    },
    "gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro",
        "model": "gemini-2.5-pro",
        "input_per_1m": 1.25,
        "audio_per_1m": None,
        "output_per_1m": 10.00,
    },
}
DEFAULT_MODEL = "gemini"

# --- Azure ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

# Web server port
PORT = int(os.getenv("PORT", "8080"))

# --- Export integrations ---
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")            # e.g. https://yourcompany.atlassian.net
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL", "")        # Atlassian account email
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN", "")        # Atlassian API token
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")       # e.g. "AO"
CONFLUENCE_PARENT_PAGE_ID = os.getenv("CONFLUENCE_PARENT_PAGE_ID", "")  # parent page ID

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")            # Notion Integration Token
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "") # Notion database to add pages to

# Set to "true" to include the full transcript in Confluence/Notion exports (default: false = summary only)
EXPORT_INCLUDE_TRANSCRIPT = os.getenv("EXPORT_INCLUDE_TRANSCRIPT", "false").lower() == "true"

# Dev/debug only: if set, shows a "Load test file" button in the UI
TEST_FILE_PATH = os.getenv("TEST_FILE_PATH", "")
