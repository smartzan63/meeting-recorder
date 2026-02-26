import os
from dotenv import load_dotenv

load_dotenv()

# OBS websocket settings
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")

# Storage directories
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "./recordings")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "./transcripts")

# Provider: "gemini" or "azure" — independent of OS
PROVIDER = os.getenv("PROVIDER", "gemini")

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MODELS = {
    "gemini": {
        "label": "Gemini 3 Flash",
        "model": "gemini-3-flash-preview",
    },
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
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
