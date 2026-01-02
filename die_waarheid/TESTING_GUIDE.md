# Die Waarheid - Comprehensive Testing Guide

## Overview

This guide covers unit testing, integration testing, and manual testing procedures for Die Waarheid.

---

## Unit Testing

### Running All Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Files

#### test_config.py
Tests configuration validation and settings.

```bash
pytest tests/test_config.py -v
```

**Test Cases:**
- Directory existence validation
- Supported formats verification
- Configuration validation
- Config summary generation
- Stress thresholds validation
- Batch size validation
- Max workers validation

#### test_chat_parser.py
Tests WhatsApp chat parsing functionality.

```bash
pytest tests/test_chat_parser.py -v
```

**Test Cases:**
- Parser initialization
- Timestamp parsing (multiple formats)
- Message type detection
- System message detection
- Sender extraction
- Metadata generation
- Statistics generation
- Message counting

#### test_forensics.py
Tests audio forensics analysis engine.

```bash
pytest tests/test_forensics.py -v
```

**Test Cases:**
- Engine initialization
- Stress level calculation bounds
- Pitch volatility bounds
- Silence ratio bounds
- Intensity calculation
- High stress detection
- Cognitive load detection
- Pitch volatility detection
- Audio processing capabilities

---

## Integration Testing

### End-to-End Workflow Test

```bash
# 1. Prepare test data
mkdir -p test_data/audio test_data/text

# 2. Add sample files
# - Add WhatsApp chat export to test_data/text/
# - Add audio files to test_data/audio/

# 3. Run integration test
python -m pytest tests/integration/ -v
```

### Manual Integration Testing

#### Test 1: Configuration Validation

```bash
python config.py
```

**Expected Output:**
```
Configuration Status: ✅ Valid
Directories: ✅ Ready
Dependencies: ✅ Installed
```

#### Test 2: Google Drive Integration

```python
from src.gdrive_handler import GDriveHandler

handler = GDriveHandler()
success, message = handler.authenticate()
print(f"Authentication: {success}")

# List files
files = handler.list_files_in_folder()
print(f"Files found: {len(files)}")
```

#### Test 3: Chat Parsing

```python
from src.chat_parser import WhatsAppParser
from pathlib import Path

parser = WhatsAppParser()
success, message = parser.parse_file(Path("data/text/chat.txt"))
print(f"Parsing: {success}")
print(f"Messages: {parser.get_message_count()}")
print(f"Metadata: {parser.get_metadata()}")
```

#### Test 4: Audio Analysis

```python
from src.forensics import ForensicsEngine
from pathlib import Path

engine = ForensicsEngine()
results = engine.analyze(Path("data/audio/sample.wav"))
print(f"Stress Level: {results['stress_level']}")
print(f"Pitch Volatility: {results['pitch_volatility']}")
```

#### Test 5: Whisper Transcription

```python
from src.whisper_transcriber import WhisperTranscriber
from pathlib import Path

transcriber = WhisperTranscriber()
result = transcriber.transcribe(Path("data/audio/sample.wav"))
print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")
```

#### Test 6: AI Analysis

```python
from src.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer()
result = analyzer.analyze_message("This is a test message")
print(f"Emotion: {result['emotion']}")
print(f"Toxicity: {result['toxicity_score']}")
```

#### Test 7: Report Generation

```python
from src.report_generator import ReportGenerator
from datetime import datetime

generator = ReportGenerator()
generator.set_case_info(
    case_id="TEST_001",
    date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
    participants=["Person A", "Person B"],
    total_messages=100,
    total_audio=10
)

success, message = generator.export_to_markdown()
print(f"Report generated: {success}")
```

#### Test 8: Streamlit UI

```bash
streamlit run app.py
```

**Manual Checks:**
- [ ] Home page loads
- [ ] Navigation works
- [ ] Data Import page functions
- [ ] Audio Analysis page displays
- [ ] Chat Analysis page works
- [ ] AI Analysis page responds
- [ ] Visualizations render
- [ ] Report Generation works
- [ ] Settings page displays
- [ ] All buttons are clickable

---

## Performance Testing

### Audio Processing Performance

```python
import time
from src.forensics import ForensicsEngine
from pathlib import Path

engine = ForensicsEngine()
audio_files = list(Path("data/audio").glob("*.wav"))

start_time = time.time()
for audio_file in audio_files:
    results = engine.analyze(audio_file)
end_time = time.time()

avg_time = (end_time - start_time) / len(audio_files)
print(f"Average processing time: {avg_time:.2f}s per file")
```

### Batch Processing Performance

```python
import time
from src.forensics import ForensicsEngine
from pathlib import Path

engine = ForensicsEngine()
audio_files = list(Path("data/audio").glob("*.wav"))

start_time = time.time()
results = engine.batch_analyze(audio_files)
end_time = time.time()

total_time = end_time - start_time
print(f"Batch processing time: {total_time:.2f}s")
print(f"Files processed: {len(results)}")
```

### Memory Usage Monitoring

```bash
# Monitor memory usage during processing
python -m memory_profiler analyze_script.py

# Or use system monitoring
# Windows
tasklist | findstr python

# macOS/Linux
ps aux | grep streamlit
```

---

## Stress Testing

### High Volume Chat Processing

```python
from src.chat_parser import WhatsAppParser
import random
from datetime import datetime, timedelta

# Generate test chat with 10,000 messages
test_messages = []
for i in range(10000):
    timestamp = datetime(2024, 1, 1) + timedelta(minutes=i)
    sender = random.choice(["Alice", "Bob"])
    text = f"Test message {i}"
    test_messages.append({
        'timestamp': timestamp,
        'sender': sender,
        'text': text
    })

# Test parsing performance
parser = WhatsAppParser()
parser.messages = test_messages

print(f"Total messages: {parser.get_message_count()}")
print(f"Metadata: {parser.get_metadata()}")
```

### Large Audio File Processing

```python
from src.forensics import ForensicsEngine
from pathlib import Path
import time

engine = ForensicsEngine()

# Test with large audio file (30+ minutes)
large_audio = Path("data/audio/long_recording.wav")

start_time = time.time()
results = engine.analyze(large_audio)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.2f}s")
print(f"Stress level: {results['stress_level']}")
```

---

## Regression Testing

### Test Suite for Previous Issues

Create `tests/test_regressions.py`:

```python
import unittest
from src.chat_parser import WhatsAppParser

class TestRegressions(unittest.TestCase):
    """Test for previously identified issues"""
    
    def test_timestamp_parsing_edge_cases(self):
        """Regression: Handle various timestamp formats"""
        parser = WhatsAppParser()
        
        formats = [
            "[01/01/2024, 10:30:45]",
            "01/01/2024, 10:30",
            "[2024-01-01 10:30:45]"
        ]
        
        for fmt in formats:
            result = parser._parse_timestamp(fmt)
            self.assertIsNotNone(result)
    
    def test_multiline_message_handling(self):
        """Regression: Handle multiline messages"""
        parser = WhatsAppParser()
        
        # Test multiline message parsing
        self.assertIsNotNone(parser)
```

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ --cov=src
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Test Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| config.py | 90% | - |
| chat_parser.py | 85% | - |
| forensics.py | 80% | - |
| ai_analyzer.py | 75% | - |
| visualizations.py | 70% | - |
| report_generator.py | 80% | - |

---

## Troubleshooting Tests

### Common Test Issues

#### ImportError: No module named 'src'

**Solution:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

#### FileNotFoundError: Test data not found

**Solution:**
```bash
# Create test data directory
mkdir -p tests/fixtures
# Add sample files to tests/fixtures/
```

#### Timeout errors

**Solution:**
```bash
# Increase timeout
pytest tests/ --timeout=300
```

#### Memory errors

**Solution:**
```bash
# Run tests with limited memory
pytest tests/ -m "not slow"
```

---

## Test Maintenance

### Regular Tasks

- **Weekly**: Run full test suite
- **Monthly**: Review test coverage
- **Quarterly**: Update test data
- **Annually**: Refactor tests

### Adding New Tests

1. Create test file in `tests/`
2. Follow naming convention: `test_*.py`
3. Use descriptive test names
4. Add docstrings
5. Run tests: `pytest tests/test_new.py -v`

---

## Performance Benchmarks

### Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Chat parsing (1000 msgs) | < 1s | Depends on message complexity |
| Audio analysis (1 min) | 5-10s | Varies by model size |
| Whisper transcription (1 min) | 10-30s | Depends on model size |
| Gemini API call | 2-5s | Network dependent |
| Report generation | < 2s | Depends on data size |

### Optimization Tips

- Use smaller Whisper model for faster processing
- Enable GPU acceleration for audio processing
- Implement caching for repeated analyses
- Use batch processing for multiple files
- Optimize database queries

---

## Support

For test-related issues:
1. Check test output for error messages
2. Review test documentation
3. Check GitHub issues
4. Run with verbose output: `pytest -vv`

---

**Last Updated**: 2024-12-29
**Version**: 1.0.0
