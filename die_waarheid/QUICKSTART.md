# Die Waarheid - Quick Start Guide

## 5-Minute Setup

### 1. Clone & Navigate
```bash
cd die_waarheid
```

### 2. Run Setup Script

**Windows:**
```bash
run.bat setup
```

**macOS/Linux:**
```bash
chmod +x run.sh
./run.sh setup
```

### 3. Configure API Keys

Edit `.env`:
```env
GEMINI_API_KEY=your_key_here
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
```

### 4. Start Application

**Windows:**
```bash
run.bat start
```

**macOS/Linux:**
```bash
./run.sh start
```

### 5. Open Browser
```
http://localhost:8501
```

---

## First Analysis

### Step 1: Import Data
1. Go to **Data Import** page
2. Upload WhatsApp chat export (`.txt`)
3. Upload audio files (`.mp3`, `.wav`, `.opus`)

### Step 2: Process Audio
1. Go to **Audio Analysis** page
2. Click **Start Audio Analysis**
3. Wait for processing to complete

### Step 3: Analyze Chat
1. Go to **Chat Analysis** page
2. Click **Analyze Chat Messages**
3. Review statistics

### Step 4: Run AI Analysis
1. Go to **AI Analysis** page
2. Click **Run AI Analysis**
3. Review findings

### Step 5: Generate Report
1. Go to **Report Generation** page
2. Enter case ID
3. Select export formats
4. Click **Generate Report**

---

## Troubleshooting

### Python Not Found
```bash
# Install Python 3.9+
# Windows: https://www.python.org/downloads/
# macOS: brew install python3
# Linux: apt-get install python3
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### API Key Error
1. Check `.env` file exists
2. Verify API key is correct
3. Restart application

### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Next Steps

- Read `README.md` for full documentation
- Check `DEPLOYMENT_GUIDE.md` for production setup
- Review `TESTING_GUIDE.md` for testing procedures
- See `IMPLEMENTATION_SUMMARY.md` for architecture details

---

**Need Help?**
- Check logs: `logs/die_waarheid.log`
- Run validation: `python config.py`
- Review documentation files
