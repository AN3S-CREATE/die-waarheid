# 🕵️ Die Waarheid - Forensic-Grade WhatsApp Communication Analysis Platform

A production-grade forensic analysis platform for WhatsApp communications, featuring advanced audio analysis, AI-powered psychological profiling, and comprehensive reporting.

## 📋 Features

- **Google Drive Integration**: OAuth 2.0 authentication for secure file access
- **Audio Forensics**: Bio-signal detection (stress, cognitive load, intensity analysis)
- **Speech Recognition**: Whisper-based transcription with Afrikaans support
- **WhatsApp Parsing**: Automated chat export processing and timeline generation
- **AI Analysis**: Gemini-powered psychological profiling and pattern detection
- **Visualizations**: Interactive Plotly charts and stress heatmaps
- **Report Generation**: Professional forensic reports with PDF/HTML export
- **Privacy**: Built-in anonymization and secure credential management

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Google Gemini API key
- HuggingFace token (optional, for speaker diarization)
- Google Drive OAuth credentials (optional, for cloud integration)

### Installation

1. **Clone and navigate to project**
```bash
cd die_waarheid
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Validate configuration**
```bash
python config.py
```

6. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
die_waarheid/
├── src/
│   ├── __init__.py
│   ├── gdrive_handler.py      # Google Drive integration
│   ├── forensics.py           # Audio analysis engine
│   ├── chat_parser.py         # WhatsApp parser
│   ├── ai_analyzer.py         # Gemini AI integration
│   ├── mobitab_builder.py     # Timeline generator
│   └── visualizations.py      # Dashboard components
├── data/
│   ├── audio/                 # Voice notes
│   ├── text/                  # Chat exports
│   ├── temp/                  # Processing temp files
│   └── output/                # Results
│       ├── mobitables/
│       ├── reports/
│       └── exports/
├── credentials/               # OAuth credentials
├── app.py                     # Main Streamlit app
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## ⚙️ Configuration

All settings are centralized in `config.py`:

- **Audio Processing**: Sample rate, Whisper model size, supported formats
- **Forensic Thresholds**: Stress detection, silence ratio, intensity spikes
- **AI Analysis**: Gemini model, temperature, token limits
- **Visualization**: Plotly theme, color schemes
- **Privacy**: Anonymization settings, temp file cleanup

### Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
HUGGINGFACE_TOKEN=your_token_here
WHISPER_MODEL_SIZE=medium
LOG_LEVEL=INFO
```

## 🔧 Development Phases

### Phase 1: Foundation ✅
- [x] Project structure and requirements
- [x] Configuration management
- [x] Environment setup

### Phase 2: Data Ingestion (In Progress)
- [ ] Google Drive handler
- [ ] Audio forensics engine
- [ ] Whisper integration

### Phase 3: Chat Processing
- [ ] WhatsApp parser
- [ ] Timeline builder

### Phase 4: AI Analysis
- [ ] Gemini integration
- [ ] Psychological profiling

### Phase 5: Visualization & Reporting
- [ ] Plotly visualizations
- [ ] Report generation

### Phase 6: Frontend & Integration
- [ ] Streamlit UI
- [ ] End-to-end testing

## 📊 Data Flow

```
Input Files (Audio/Chat)
    ↓
[Google Drive Handler] → Download & validate
    ↓
├─→ [Chat Parser] → WhatsApp timeline
│       ↓
│   [Mobitab Builder] → Structured messages
│
└─→ [Forensics Engine] → Audio analysis
        ↓
    [Whisper] → Transcription
        ↓
    Bio-signals (stress, silence, intensity)
    
Both streams merge ↓
[AI Analyzer] → Gemini analysis
    ├─ Psychological profile
    ├─ Contradiction detection
    ├─ Toxicity patterns
    └─ Trust score
    
    ↓
[Visualizations] → Charts & graphs
    ↓
[Report Template] → Markdown/PDF/HTML
    ↓
[Streamlit UI] → Display & export
```

## 🔐 Security

- **Credentials**: All API keys stored in `.env` (not committed)
- **Data**: Automatic temp file cleanup after 24 hours
- **Privacy**: Optional name/phone anonymization
- **Logging**: Structured logging with file rotation

## 📝 Logging

Logs are written to `die_waarheid.log` with configurable level:

```python
# In config.py
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 🧪 Testing

Run configuration validation:
```bash
python config.py
```

## 📄 License

Proprietary - AN3S Workspace

## 👤 Author

AN3S Workspace

## 📞 Support

For issues or questions, refer to the troubleshooting section in the specification document.

---

**Version**: 1.0.0  
**Status**: Development (Phase 1 Complete)
