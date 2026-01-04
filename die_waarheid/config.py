"""
Configuration Settings for Die Waarheid
Central configuration management for all application settings
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TEXT_DIR = DATA_DIR / "text"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
MOBITAB_DIR = OUTPUT_DIR / "mobitables"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPORTS_DIR = OUTPUT_DIR / "exports"
CREDENTIALS_DIR = BASE_DIR / "credentials"

for directory in [DATA_DIR, AUDIO_DIR, TEXT_DIR, TEMP_DIR, OUTPUT_DIR,
                  MOBITAB_DIR, REPORTS_DIR, EXPORTS_DIR, CREDENTIALS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# API CREDENTIALS
# ==============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "") or os.getenv("HUGGINGFACE_API_KEY", "")
GOOGLE_CREDENTIALS_PATH = CREDENTIALS_DIR / "google_credentials.json"
TOKEN_PICKLE_PATH = CREDENTIALS_DIR / "token.pickle"

# ==============================================================================
# GOOGLE DRIVE SETTINGS
# ==============================================================================

GDRIVE_AUDIO_FOLDER = "Investigation_Audio"
GDRIVE_TEXT_FOLDER = "Investigation_Text"
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# ==============================================================================
# AUDIO PROCESSING SETTINGS
# ==============================================================================

SUPPORTED_AUDIO_FORMATS = ['.opus', '.ogg', '.mp3', '.wav', '.m4a', '.aac']
TARGET_SAMPLE_RATE = 16000

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")

WHISPER_OPTIONS = {
    "language": "af",
    "task": "transcribe",
    "best_of": 5,
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

# ==============================================================================
# FORENSIC ANALYSIS THRESHOLDS
# ==============================================================================

STRESS_THRESHOLD_HIGH = 50
SILENCE_RATIO_THRESHOLD = 0.4
INTENSITY_SPIKE_THRESHOLD = 0.7

STRESS_WEIGHTS = {
    'pitch': 0.35,
    'silence': 0.20,
    'intensity': 0.25,
    'mfcc': 0.20
}

# Thresholds for pattern detection
GASLIGHTING_THRESHOLD = 0.3
TOXICITY_THRESHOLD = 0.4
NARCISSISTIC_PATTERN_THRESHOLD = 0.35

GASLIGHTING_PHRASES = [
    "never said that", "you're crazy", "imagining things",
    "overreacting", "too sensitive", "making things up",
    "didn't happen", "in your head", "being dramatic",
    "jy dink te veel", "jy is mal", "dit het nooit gebeur nie"
]

TOXICITY_PHRASES = [
    "stupid", "hate", "idiot", "shut up", "dumb", "useless",
    "pathetic", "loser", "worthless", "disgusting",
    "dom", "haat", "idioot", "hou jou bek", "nutteloos"
]

NARCISSISTIC_PATTERNS = [
    "I'm the victim", "you made me", "look what you made me do",
    "everyone agrees with me", "nobody likes you",
    "ek is die slagoffer", "jy het my gemaak"
]

# ==============================================================================
# MOBITAB SETTINGS
# ==============================================================================

MOBITAB_COLUMNS = [
    "Index",
    "Msg_ID",
    "Sender",
    "Recorded_At",
    "Speaker_Count",
    "Transcript",
    "Tone_Emotion",
    "Pitch_Volatility",
    "Silence_Ratio",
    "Intensity_Max",
    "Forensic_Flag"
]

MOBITAB_FILENAME = "timeline.md"

# ==============================================================================
# AI ANALYSIS SETTINGS
# ==============================================================================

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_TOKENS = 8192

GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ==============================================================================
# VISUALIZATION SETTINGS
# ==============================================================================

PLOTLY_THEME = "plotly_dark"
STRESS_HEATMAP_COLORSCALE = "RdYlGn_r"

SPEAKER_COLORS = {
    "SPEAKER_00": "#1f77b4",
    "SPEAKER_01": "#ff7f0e",
    "SPEAKER_02": "#2ca02c",
    "SPEAKER_03": "#d62728",
    "SPEAKER_04": "#9467bd",
}

# ==============================================================================
# BATCH PROCESSING SETTINGS
# ==============================================================================

BATCH_SIZE = 20
MAX_WORKERS = 4

# ==============================================================================
# PRIVACY & SECURITY
# ==============================================================================

ANONYMIZE_PHONE_NUMBERS = True
ANONYMIZE_NAMES = True
AUTO_CLEANUP_TEMP_FILES = True
CLEANUP_DELAY_HOURS = 24

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "die_waarheid.log"

# Create logs directory
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

REPORT_TEMPLATE = """
# 🕵️ DIE WAARHEID FORENSIC ANALYSIS REPORT

**Generated:** {timestamp}

**Case ID:** {case_id}

**Analysis Period:** {date_range}

---

## 📊 EXECUTIVE SUMMARY

**Total Messages Analyzed:** {total_messages}

**Voice Notes Analyzed:** {total_audio}

**Participants:** {participants}

**Analysis Duration:** {duration}

### Trust Score: {trust_score}/100

{trust_interpretation}

---

## 🚨 CRITICAL FINDINGS

{critical_findings}

---

## 📈 COMMUNICATION METRICS

{metrics_summary}

---

## 🎭 PSYCHOLOGICAL PROFILE

{psychological_profile}

---

## ⚠️ DETECTED CONTRADICTIONS

{contradictions}

---

## 🔬 BIO-SIGNAL ANALYSIS

{biosignal_summary}

---

## 📝 RECOMMENDATIONS

{recommendations}

---

## ⚖️ LEGAL DISCLAIMER

This report is generated by AI-assisted analysis and should not be considered
as sole evidence in legal proceedings. All findings should be verified by
qualified professionals. The accuracy of voice stress analysis is subject
to debate in the forensic community.

**Report ID:** {report_id}
"""

# ==============================================================================
# EXPORT SETTINGS
# ==============================================================================

EXPORT_FORMATS = ["markdown", "pdf", "html", "json", "csv"]
DEFAULT_EXPORT_FORMAT = "markdown"

# ==============================================================================
# APPLICATION METADATA
# ==============================================================================

APP_NAME = "Die Waarheid"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Forensic-Grade WhatsApp Communication Analysis Platform"
AUTHOR = "AN3S Workspace"

# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_config():
    """Validate critical configuration settings"""
    errors = []
    warnings = []

    if not GEMINI_API_KEY:
        errors.append("❌ GEMINI_API_KEY not set in .env file")

    if not HUGGINGFACE_TOKEN:
        warnings.append("⚠️ HUGGINGFACE_TOKEN not set (speaker diarization will be disabled)")

    if not GOOGLE_CREDENTIALS_PATH.exists():
        warnings.append("⚠️ Google Drive credentials not found (manual upload only)")

    return errors, warnings


def get_config_summary():
    """Get a summary of current configuration"""
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "whisper_model": WHISPER_MODEL_SIZE,
        "gemini_model": GEMINI_MODEL,
        "log_level": LOG_LEVEL,
        "max_workers": MAX_WORKERS,
        "batch_size": BATCH_SIZE,
    }


if __name__ == "__main__":
    errors, warnings = validate_config()
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not errors and not warnings:
        print("✅ Configuration is valid")
    
    print("\nConfiguration Summary:")
    import json
    print(json.dumps(get_config_summary(), indent=2))
