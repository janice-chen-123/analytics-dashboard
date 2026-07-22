"""
config.py — Project-wide constants and environment variable loading.

All configuration values are read here. Other modules import from this file
instead of calling os.getenv() directly.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE = "AI Data Analyst Dashboard"
APP_SUBTITLE = "Upload a CSV or Excel file, explore your data, and get AI-generated business insights."
APP_VERSION = "1.0.0"

# ── File upload limits ────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# ── SQLite ────────────────────────────────────────────────────────────────────
DATABASE_PATH = "database/analytics.db"
DEFAULT_TABLE_NAME = "sales_data"

# ── AI defaults ───────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
TOP_N_CATEGORIES = 10


def get_api_key() -> str | None:
    """
    Return the OpenAI API key.

    Checks Streamlit Secrets first (for cloud deployment), then falls back
    to the environment variable (for local development via .env).

    Returns None if the key is not configured — callers must handle this.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY") or None
