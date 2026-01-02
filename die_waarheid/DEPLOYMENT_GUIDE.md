# 🚀 Die Waarheid - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Configuration](#configuration)
4. [Running Tests](#running-tests)
5. [Production Deployment](#production-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 10GB minimum (for models and data)
- **GPU**: Optional (NVIDIA CUDA for faster processing)

### Required Accounts
- **Google Cloud Console**: For Google Drive API access
- **Gemini API**: For AI analysis (free tier available)
- **HuggingFace** (optional): For advanced NLP features

---

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd die_waarheid
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python config.py
```

Expected output:
```
Configuration Status: ✅ Valid
Directories: ✅ Ready
Dependencies: ✅ Installed
```

---

## Configuration

### 1. Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
# Google Drive API
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
GOOGLE_CREDENTIALS_PATH=credentials/credentials.json

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here

# HuggingFace (optional)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Application Settings
APP_DEBUG=False
LOG_LEVEL=INFO
MAX_WORKERS=4
BATCH_SIZE=20
```

### 2. Google Drive Setup

#### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project: "Die Waarheid"
3. Enable Google Drive API

#### Step 2: Create OAuth Credentials
1. Go to Credentials → Create Credentials → OAuth 2.0 Client ID
2. Choose "Desktop application"
3. Download JSON file
4. Save as `credentials/credentials.json`

#### Step 3: Get Folder ID
1. Open your Google Drive folder
2. Copy the folder ID from URL: `https://drive.google.com/drive/folders/{FOLDER_ID}`
3. Add to `.env`: `GOOGLE_DRIVE_FOLDER_ID={FOLDER_ID}`

### 3. Gemini API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=your_key_here`

### 4. Directory Structure

Verify directories exist:
```
die_waarheid/
├── data/
│   ├── audio/          # Voice notes
│   ├── text/           # Chat exports
│   ├── temp/           # Processing files
│   └── output/
│       ├── mobitables/ # Generated timelines
│       ├── reports/    # Generated reports
│       └── exports/    # Exported data
├── credentials/        # OAuth credentials
├── logs/              # Application logs
└── tests/             # Test suite
```

Create directories if missing:
```bash
mkdir -p data/{audio,text,temp,output/{mobitables,reports,exports}}
mkdir -p credentials logs tests
```

---

## Running Tests

### Unit Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_config.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Configuration Test
```bash
python config.py
```

### Manual Testing Checklist

- [ ] Configuration validation passes
- [ ] Google Drive authentication works
- [ ] Chat file parsing works
- [ ] Audio file processing works
- [ ] Whisper transcription works
- [ ] Gemini API calls work
- [ ] Report generation works
- [ ] Streamlit UI loads
- [ ] All pages render correctly
- [ ] Data export functions work

---

## Production Deployment

### 1. Pre-Deployment Checklist

```bash
# Verify all tests pass
python -m pytest tests/ -v

# Validate configuration
python config.py

# Check dependencies
pip list

# Verify directories
ls -la data/
```

### 2. Environment Setup

```bash
# Set production environment
export FLASK_ENV=production
export APP_DEBUG=False
export LOG_LEVEL=WARNING
```

### 3. Run Application

```bash
# Start Streamlit server
streamlit run app.py --logger.level=warning

# Or with custom configuration
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --logger.level warning
```

### 4. Access Application

Open browser and navigate to:
```
http://localhost:8501
```

### 5. Production Considerations

#### Security
- [ ] Use HTTPS/SSL certificates
- [ ] Implement authentication layer
- [ ] Restrict API access
- [ ] Enable audit logging
- [ ] Regular security updates

#### Performance
- [ ] Use caching layer (Redis)
- [ ] Implement database backend
- [ ] Enable async processing
- [ ] Monitor resource usage
- [ ] Set up load balancing

#### Monitoring
- [ ] Set up logging aggregation
- [ ] Configure alerts
- [ ] Monitor API usage
- [ ] Track error rates
- [ ] Performance metrics

---

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/{audio,text,temp,output/{mobitables,reports,exports}} \
    && mkdir -p credentials logs

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  die-waarheid:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./credentials:/app/credentials
      - ./logs:/app/logs
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GOOGLE_DRIVE_FOLDER_ID=${GOOGLE_DRIVE_FOLDER_ID}
      - APP_DEBUG=False
    restart: unless-stopped
```

### 3. Build and Run

```bash
# Build image
docker build -t die-waarheid:latest .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/credentials:/app/credentials \
  -e GEMINI_API_KEY=your_key \
  die-waarheid:latest

# Or use docker-compose
docker-compose up -d
```

---

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Google Drive Authentication Fails
```
Error: credentials.json not found
```

**Solution:**
1. Download credentials from Google Cloud Console
2. Save to `credentials/credentials.json`
3. Restart application

#### 3. Gemini API Errors
```
Error: Invalid API key
```

**Solution:**
1. Verify API key in `.env`
2. Check API is enabled in Google Cloud Console
3. Regenerate key if needed

#### 4. Audio Processing Fails
```
Error: librosa could not load audio file
```

**Solution:**
```bash
# Install ffmpeg
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg
# Linux: apt-get install ffmpeg
```

#### 5. Memory Issues
```
MemoryError: Unable to allocate memory
```

**Solution:**
- Reduce batch size in config.py
- Process files individually
- Increase system RAM
- Use smaller Whisper model

#### 6. Slow Performance
```
Analysis taking too long
```

**Solution:**
- Use smaller Whisper model (tiny/base)
- Reduce audio sample rate
- Enable GPU acceleration
- Increase MAX_WORKERS

### Debug Mode

Enable debug logging:
```bash
# In .env
APP_DEBUG=True
LOG_LEVEL=DEBUG

# Or via command line
streamlit run app.py --logger.level=debug
```

Check logs:
```bash
tail -f logs/die_waarheid.log
```

---

## Monitoring & Maintenance

### Log Files

Location: `logs/die_waarheid.log`

View logs:
```bash
# Last 100 lines
tail -100 logs/die_waarheid.log

# Follow logs in real-time
tail -f logs/die_waarheid.log

# Search for errors
grep ERROR logs/die_waarheid.log
```

### Cleanup Temporary Files

```bash
# Manual cleanup
rm -rf data/temp/*

# Automatic cleanup (configured in config.py)
# Runs every 24 hours by default
```

### Performance Monitoring

```bash
# Monitor resource usage
# Windows
tasklist | findstr python

# macOS/Linux
ps aux | grep streamlit
```

### Database Maintenance

If using database backend:
```bash
# Backup database
cp database.db database.db.backup

# Optimize database
sqlite3 database.db "VACUUM;"
```

### Regular Maintenance Tasks

- [ ] **Daily**: Check error logs
- [ ] **Weekly**: Verify backups
- [ ] **Monthly**: Update dependencies
- [ ] **Quarterly**: Security audit
- [ ] **Annually**: Full system review

### Update Dependencies

```bash
# Check for updates
pip list --outdated

# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade streamlit
```

---

## Performance Tuning

### Optimize Whisper

```python
# Use smaller model for faster processing
WHISPER_MODEL_SIZE = "base"  # Instead of "medium"

# Or use quantized version
WHISPER_QUANTIZED = True
```

### Optimize Gemini

```python
# Reduce token limit for faster responses
GEMINI_MAX_TOKENS = 2048  # Instead of 8192

# Increase temperature for faster inference
GEMINI_TEMPERATURE = 0.5  # Instead of 0.2
```

### Enable Caching

```python
# In config.py
ENABLE_CACHE = True
CACHE_TTL = 3600  # 1 hour
```

### Parallel Processing

```python
# Increase workers for faster batch processing
MAX_WORKERS = 8  # Adjust based on CPU cores
BATCH_SIZE = 50  # Increase for better throughput
```

---

## Backup & Recovery

### Backup Strategy

```bash
# Backup important files
tar -czf die_waarheid_backup.tar.gz \
  credentials/ \
  data/output/ \
  config.py \
  .env

# Backup to cloud
aws s3 cp die_waarheid_backup.tar.gz s3://my-bucket/
```

### Recovery Procedure

```bash
# Restore from backup
tar -xzf die_waarheid_backup.tar.gz

# Verify restoration
python config.py

# Restart application
streamlit run app.py
```

---

## Support & Resources

- **Documentation**: See README.md
- **Issues**: Check IMPLEMENTATION_SUMMARY.md
- **Testing**: Run `pytest tests/`
- **Logs**: Check `logs/die_waarheid.log`
- **Configuration**: Edit `config.py`

---

**Last Updated**: 2024-12-29  
**Version**: 1.0.0  
**Status**: Production Ready
