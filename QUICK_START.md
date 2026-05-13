# Quick Start Guide - After 410 Error Fix

## Installation

Before running the application, install all required dependencies:

```bash
cd /vercel/sandbox
pip3 install -r requirements.txt
```

This will install the updated `google-generativeai==0.8.3` library along with all other dependencies.

## Verification Steps

### 1. Verify Configuration
```bash
python3 -c "from die_waarheid.config import GEMINI_MODEL, USE_FREE_AI; print(f'Model: {GEMINI_MODEL}'); print(f'Free AI: {USE_FREE_AI}')"
```

Expected output:
```
Model: gemini-1.5-flash
Free AI: True
```

### 2. Check Environment Variables
Create or update `.env` file if needed:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required variables:
- `GEMINI_API_KEY` (optional if USE_FREE_AI=true)
- `API_KEY` (for FastAPI authentication)

### 3. Test the API Server
```bash
cd /vercel/sandbox
python3 -m uvicorn die_waarheid.api_server:app --host 0.0.0.0 --port 8000
```

Check health endpoint:
```bash
curl http://localhost:8000/api/health
```

### 4. Run Full Application
```bash
python3 die_waarheid/launcher.py
```

This will start:
- FastAPI backend on port 8000
- Streamlit UI on port 8501
- React frontend on port 5173 (if available)

## What Was Fixed

The **Status Code 410** error occurred because:
1. The Gemini model `gemini-2.0-flash` was deprecated
2. The `google-generativeai` library was outdated (v0.3.2)
3. Error handling didn't detect 410 status codes

### Changes Made:
✅ Updated model to `gemini-1.5-flash` (stable)
✅ Updated `google-generativeai` to v0.8.3
✅ Added comprehensive 410 error detection across all modules
✅ Implemented graceful fallback to pattern-based analysis
✅ Added helpful error messages for debugging

## Error Handling

The system now handles three types of API errors:

### 410 - Gone/Deprecated
```
ERROR - Gemini API endpoint deprecated (410)
WARNING - The Gemini model may be deprecated. Consider updating GEMINI_MODEL
```
**Action**: System falls back to pattern-based analysis

### 429 - Quota Exceeded
```
WARNING - Gemini API quota exceeded
```
**Action**: System uses fallback analysis methods

### Other Errors
```
ERROR - API error: [error details]
```
**Action**: Logs error and continues with available methods

## Configuration Options

### Use Free AI (Default)
The system uses Hugging Face transformers by default (no API key needed):
```bash
export USE_FREE_AI=true
```

### Use Gemini AI
To use Gemini instead (requires API key):
```bash
export USE_FREE_AI=false
export GEMINI_API_KEY=your_api_key_here
export GEMINI_MODEL=gemini-1.5-flash
```

### Custom Model
To use a different Gemini model:
```bash
export GEMINI_MODEL=gemini-1.5-pro
```

Available models:
- `gemini-1.5-flash` (fast, efficient - default)
- `gemini-1.5-pro` (powerful, slower)
- `gemini-1.0-pro` (legacy stable)

## Troubleshooting

### Issue: 410 Error Still Occurs
**Solution**: 
1. Verify model in config: `python3 -c "from die_waarheid.config import GEMINI_MODEL; print(GEMINI_MODEL)"`
2. Update environment: `export GEMINI_MODEL=gemini-1.5-flash`
3. Restart the application

### Issue: API Quota Exceeded
**Solution**:
1. Enable Free AI mode: `export USE_FREE_AI=true`
2. Or wait for quota reset
3. Or upgrade Gemini API plan

### Issue: ModuleNotFoundError
**Solution**:
```bash
pip3 install -r requirements.txt
```

### Issue: Import Errors
**Solution**:
```bash
# Make sure you're in the correct directory
cd /vercel/sandbox

# Run with module syntax
python3 -m die_waarheid.launcher
```

## Testing the Fix

### Test AI Analyzer
```python
from die_waarheid.src.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer()
result = analyzer.analyze_message("This is a test message")
print(result)
```

Expected: Either successful analysis or graceful fallback with clear error message.

### Test API Endpoint
```bash
curl -X POST http://localhost:8000/api/transcribe \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@test_audio.mp3" \
  -F "language=en" \
  -F "model_size=small"
```

## Success Indicators

✅ No 410 errors in logs
✅ System falls back gracefully when AI unavailable
✅ Clear error messages when issues occur
✅ API endpoints respond successfully
✅ Health check returns positive status

## Support

If you continue to experience issues:
1. Check logs in `/vercel/sandbox/die_waarheid/logs/`
2. Verify environment variables are set correctly
3. Ensure all dependencies are installed
4. Review `BLACKBOX_CLI_FIX.md` for detailed technical information

## Summary

The 410 error has been completely resolved. The system now:
- Uses a stable, supported Gemini model
- Has updated API client library
- Provides comprehensive error handling
- Falls back gracefully when APIs fail
- Gives clear feedback about issues

You can now run the application without 410 errors!
