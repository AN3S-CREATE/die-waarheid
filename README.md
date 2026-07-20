# Die Waarheid - Advanced Forensic Analysis System

> CI contributor guide: [docs/ci.md](docs/ci.md)

🕵️ **Forensic-grade WhatsApp communication analysis platform with AI-powered insights**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-blue.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- 🎙️ **Audio Forensics**: Bio-signal detection, stress analysis, speaker identification
- 💬 **Chat Analysis**: WhatsApp message parsing, pattern detection, timeline reconstruction  
- 🧠 **AI Analysis**: Gemini-powered psychological profiling and insight generation
- 📊 **Visualizations**: Interactive charts, timelines, and relationship graphs
- 📄 **Report Generation**: Multi-format forensic reports (PDF, Excel, HTML)
- 🔒 **Security**: API authentication, rate limiting, secure file handling
- ⚡ **Performance**: Caching, async operations, optimized processing

## 🚀 Quick Start

> **⚠️ IMPORTANT**: If you're experiencing **"Status code 410"** errors, see [BLACKBOX_CLI_FIX.md](BLACKBOX_CLI_FIX.md) for the solution. The issue has been fixed by updating to `gemini-1.5-flash` model.

### Prerequisites

- **Python 3.11+** (3.11 recommended)
- **Node.js 18+** (for React frontend)
- **8GB+ RAM** (16GB recommended for large analyses)
- **Optional**: CUDA-capable GPU for faster processing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AN3S-CREATE/die-waarheid.git
cd die-waarheid
```

2. **Install Python dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

4. **Configure environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# GEMINI_API_KEY=your_google_api_key_here
# API_KEY=generate_secure_key_here
```

5. **Initialize database**
```bash
python -c "from die_waarheid.src.database import init_database; init_database()"
```

### Running the Application

**🎯 Option 1: Unified Launcher (Recommended)**
```bash
python die_waarheid/launcher.py
```

This automatically starts:
- 📊 **Streamlit UI**: http://localhost:8501
- 🔌 **FastAPI Backend**: http://localhost:8000  
- 📱 **React Frontend**: http://localhost:5173

**⚙️ Option 2: Manual (Development)**
```bash
# Terminal 1: Start API server
python -m uvicorn die_waarheid.api_server:app --reload --port 8000

# Terminal 2: Start Streamlit
streamlit run die_waarheid/app.py --server.port 8501

# Terminal 3: Start React frontend  
cd frontend && npm run dev
```

## 📖 Usage Guide

### 1. **Upload Data**
- Import WhatsApp chat exports (.txt files)
- Upload voice notes and audio files
- Supported formats: MP3, WAV, OPUS, OGG, M4A, AAC

### 2. **Train Speakers**
- Upload voice samples for each participant
- System learns voice fingerprints for identification
- Minimum 3 samples per speaker recommended

### 3. **Analyze Communications**
- Run automated forensic analysis pipeline
- Audio: Stress detection, pitch analysis, bio-signals
- Text: Pattern detection, timeline reconstruction
- AI: Psychological profiling, relationship analysis

### 4. **Review Results**
- Interactive dashboards with charts and timelines
- Chronological analysis table with all findings
- Speaker identification and confidence scores
- Stress level indicators and anomaly detection

### 5. **Generate Reports**
- Comprehensive forensic reports in multiple formats
- Evidence scoring and investigative checklists
- Export data for further analysis

## 🏗️ Architecture

```
die-waarheid/
├── die_waarheid/           # Python backend
│   ├── app.py             # Streamlit web application
│   ├── api_server.py      # FastAPI REST API
│   ├── launcher.py        # Unified service launcher
│   ├── config.py          # Configuration management
│   ├── src/               # Core modules
│   │   ├── forensics.py   # Audio forensics engine
│   │   ├── chat_parser.py # WhatsApp parser
│   │   ├── ai_analyzer.py # Gemini integration
│   │   ├── speaker_identification.py
│   │   ├── visualizations.py
│   │   └── ...
│   └── tests/             # Test suite
├── frontend/              # React TypeScript frontend
│   ├── src/
│   │   ├── pages/         # React pages
│   │   ├── components/    # UI components
│   │   └── services/      # API integration
│   └── package.json
├── data/                  # Data directories
│   ├── audio/            # Audio files
│   ├── text/             # Chat files
│   └── output/           # Analysis results
└── requirements.txt
```

## 🔌 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/transcribe` - Transcribe audio to text
- `POST /api/analyze` - Perform forensic audio analysis  
- `GET /api/speakers` - Get speaker profiles
- `POST /api/speakers/train` - Train speaker identification
- `GET /api/health` - System health check

## 🔒 Security Features

- **API Authentication**: Bearer token authentication
- **Rate Limiting**: Prevents abuse and DoS attacks
- **File Validation**: Size limits and format checking
- **Secure CORS**: Restricted origins and headers
- **Input Validation**: Pydantic models for all requests
- **Temp File Cleanup**: Automatic cleanup of temporary files

## ⚡ Performance Features

- **Model Caching**: Whisper models cached in memory
- **Database Pooling**: Connection pooling for better performance
- **Async Operations**: Non-blocking file operations
- **Result Caching**: Analysis results cached for reuse
- **GPU Support**: Automatic CUDA detection and usage

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=die_waarheid --cov-report=html

# Run specific test category
pytest tests/test_forensics.py -v
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t die-waarheid .
docker run -p 8000:8000 -p 8501:8501 die-waarheid
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Required
GEMINI_API_KEY=your_google_api_key
API_KEY=your_secure_api_key

# Optional
WHISPER_MODEL_SIZE=small
USE_GPU=true
DATABASE_URL=sqlite:///./die_waarheid.db
ENVIRONMENT=development
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. API Connection Issues**
```bash
# Check if API server is running
curl http://localhost:8000/api/health
```

**3. Model Loading Errors**
```bash
# Clear model cache
rm -rf ~/.cache/whisper/
```

**4. Database Issues**
```bash
# Reinitialize database
python -c "from die_waarheid.src.database import init_database; init_database()"
```

### Getting Help

- 📧 **Email**: support@an3s-workspace.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/AN3S-CREATE/die-waarheid/issues)
- 📖 **Documentation**: [Wiki](https://github.com/AN3S-CREATE/die-waarheid/wiki)

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Google Gemini** for AI analysis
- **Pyannote** for speaker diarization
- **Streamlit** for rapid UI development
- **FastAPI** for high-performance API

---

**⚖️ Legal Notice**: This application is provided for forensic analysis purposes. All findings should be verified by qualified professionals. Users are responsible for compliance with applicable laws and regulations.

**🔒 Privacy**: All data is processed locally. No data is transmitted to external services except for AI analysis (Gemini API) when explicitly requested.
