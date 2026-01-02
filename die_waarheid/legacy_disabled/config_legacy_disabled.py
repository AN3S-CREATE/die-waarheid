"""
Configuration Settings for Die Waarheid

Central configuration management for the complete forensic application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
CREDENTIALS_DIR = BASE_DIR / "credentials"
AUDIO_DIR = DATA_DIR / "audio"
TEXT_DIR = DATA_DIR / "text"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
MOBITABLES_DIR = OUTPUT_DIR / "mobitables"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPORTS_DIR = OUTPUT_DIR / "exports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CREDENTIALS_DIR, AUDIO_DIR, TEXT_DIR, TEMP_DIR, 
                 OUTPUT_DIR, MOBITABLES_DIR, REPORTS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Google Drive Configuration
GOOGLE_CREDENTIALS_FILE = CREDENTIALS_DIR / "google_credentials.json"
GOOGLE_TOKEN_FILE = CREDENTIALS_DIR / "token.pickle"
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Audio Processing
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_LENGTH = 30 * 60  # 30 minutes max

# Forensic Analysis
STRESS_THRESHOLD_HIGH = 70
STRESS_THRESHOLD_MEDIUM = 50
STRESS_THRESHOLD_LOW = 30

# Visualization
THEME_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Export Settings
EXPORT_FORMATS = ['csv', 'excel', 'pdf', 'json']
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance
MAX_CONCURRENT_AUDIO_PROCESSES = 2
CHUNK_SIZE = 1000  # For large datasets

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "logs" / "die_waarheid.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': '🔬 DIE WAARHEID: Forensic WhatsApp Analysis',
    'page_icon': '🔬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Cache Settings
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
