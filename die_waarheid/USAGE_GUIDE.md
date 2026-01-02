# Die Waarheid - Usage Guide

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_api_key_here"  # Optional but recommended
```

### 2. Run the System

There are **3 main ways** to use Die Waarheid:

---

## Method 1: Main Orchestrator (Complete Analysis)

**Best for**: Full forensic analysis workflow

```python
from src.main_orchestrator import MainOrchestrator

# Initialize the system
orchestrator = MainOrchestrator()

# Create investigation case
orchestrator.create_case("CASE_001", "Investigation Example")
```

# Add evidence (text files, audio files, chat exports)
orchestrator.add_evidence("EV_001", "chat_export", "chat.txt", "WhatsApp export")
orchestrator.add_evidence("EV_002", "audio_statement", "statement.wav", "Witness interview")

# Run complete 12-stage analysis
results = orchestrator.run_complete_analysis()

# Export results
orchestrator.export_results(results, "analysis_results.json")

print("Analysis complete! Check analysis_results.json")
```

---

## Method 2: Integration Orchestrator (Modular Analysis)

**Best for**: Individual module testing and specific analysis

```python
from src.integration_orchestrator import IntegrationOrchestrator
```

# Initialize
orchestrator = IntegrationOrchestrator()

# Create case
case_id = orchestrator.create_case("CASE_002", "Test Investigation")

# Add evidence
evidence_id = orchestrator.add_evidence(case_id, "sample.txt")

# Run specific analysis stages
alerts = orchestrator.run_alert_analysis(case_id)
scores = orchestrator.run_evidence_scoring(case_id)
checklist = orchestrator.run_checklist_generation(case_id)

print(f"Alerts: {len(alerts)} found")
print(f"Evidence scores: {len(scores)} items")
print(f"Checklist items: {len(checklist)} actions")
```

---

## Method 3: Individual Modules (Direct Usage)

**Best for**: Specific forensic tasks

### Alert System
```python
from src.alert_system import AlertSystem

alerts = AlertSystem()
result = alerts.check_contradiction(
    statement1="I was at home all night",
    statement2="I went to the store at 10 PM"
)
if result.contradiction_detected:
    print(f"Contradiction found: {result.reason}")
```

### Evidence Scoring
```python
from src.evidence_scoring import EvidenceScoringSystem

scorer = EvidenceScoringSystem()
score = scorer.calculate_evidence_score(
    text="I saw the suspect running away",
    source="eyewitness",
    timestamp="2025-01-01 10:00:00"
)
print(f"Evidence strength: {score.overall_score}/100")
```

### Comparative Psychology
```python
from src.comparative_psychology import ComparativePsychologyAnalyzer

analyzer = ComparativePsychologyAnalyzer()
comparison = analyzer.compare_participants(
    participant1_statements=["I didn't do it", "I was innocent"],
    participant2_statements=["He did it", "I saw him"]
)
print(f"Agreement level: {comparison.agreement_level}")
```

---

## Method 4: Command Line Interface

**Best for**: Quick analysis from terminal

```bash
# Run integration tests
python test_integration.py

# Run health check
python -m src.health

# Run main orchestrator
python -m src.main_orchestrator
```

---

## Method 5: Web Interface (Streamlit)

**Best for**: Visual, interactive analysis

```bash
# Start web interface
streamlit run src/web_interface.py
```

Then open http://localhost:8501 in your browser.

---

## Input Data Formats

### Text Evidence
```python
# Plain text files
orchestrator.add_evidence("EV_001", "text", "statement.txt", "Witness statement")

# Chat exports (WhatsApp, SMS, etc.)
orchestrator.add_evidence("EV_002", "chat_export", "chat.txt", "WhatsApp conversation")

# Transcripts
orchestrator.add_evidence("EV_003", "transcript", "interview.txt", "Police interview")
```

### Audio Evidence
```python
# Audio files (WAV, MP3, M4A)
orchestrator.add_evidence("EV_004", "audio", "statement.wav", "Voice recording")

# Multiple audio files
orchestrator.add_evidence("EV_005", "audio", "suspect1.wav", "Suspect interview")
orchestrator.add_evidence("EV_006", "audio", "suspect2.wav", "Witness interview")
```

### Mixed Evidence
```python
# Combine text and audio
orchestrator.add_evidence("EV_007", "mixed", "evidence.zip", "Complete case file")
```

---

## Output Formats

### JSON Export
```python
# Complete analysis results
results = orchestrator.run_complete_analysis()
orchestrator.export_results(results, "results.json")

# Individual module results
alerts = orchestrator.run_alert_analysis(case_id)
with open("alerts.json", "w") as f:
    json.dump(alerts, f, indent=2)
```

### HTML Reports
```python
# Timeline visualization
timeline = orchestrator.run_timeline_analysis(case_id)
timeline.export_html("timeline.html")

# Contradiction timeline
contradictions = orchestrator.run_contradiction_analysis(case_id)
contradictions.export_html("contradictions.html")
```

### CSV Export
```python
# Evidence scores
import pandas as pd
scores = orchestrator.run_evidence_scoring(case_id)
df = pd.DataFrame([s.__dict__ for s in scores])
df.to_csv("evidence_scores.csv", index=False)
```

---

## Configuration Options

### Environment Variables
```bash
# Google Generative AI (optional)
export GEMINI_API_KEY="your_api_key"
export GEMINI_MODEL="gemini-pro"
export GEMINI_TEMPERATURE="0.7"

# Database (optional)
export DATABASE_URL="sqlite:///die_waarheid.db"

# Performance
export CACHE_ENABLED="true"
export MAX_WORKERS="4"
```

### Python Configuration
```python
from src.config import (
    GEMINI_API_KEY,
    GASLIGHTING_THRESHOLD,
    TOXICITY_THRESHOLD,
    NARCISSISTIC_PATTERN_THRESHOLD
)

# Adjust thresholds
GASLIGHTING_THRESHOLD = 0.8  # Higher = more strict
TOXICITY_THRESHOLD = 0.6     # Lower = more sensitive
```

---

## Example Workflows

### Workflow 1: Witness Statement Analysis
```python
# 1. Create case
orchestrator.create_case("WITNESS_001", "Witness Credibility Assessment")

# 2. Add witness statements
orchestrator.add_evidence("W1", "text", "witness1_statement.txt", "Primary witness")
orchestrator.add_evidence("W2", "text", "witness2_statement.txt", "Secondary witness")

# 3. Run analysis
results = orchestrator.run_complete_analysis()

# 4. Review contradictions
contradictions = results.get("contradiction_timeline", [])
for contradiction in contradictions:
    print(f"Contradiction: {contradiction.description}")

# 5. Check evidence scores
scores = results.get("evidence_scoring", [])
for score in scores:
    print(f"Evidence {score.evidence_id}: {score.overall_score}/100")
```

### Workflow 2: Multi-Party Investigation
```python
# 1. Setup
orchestrator.create_case("MULTI_001", "Multi-Party Investigation")

# 2. Add all evidence
orchestrator.add_evidence("P1", "audio", "person1.wav", "Suspect statement")
orchestrator.add_evidence("P2", "audio", "person2.wav", "Witness statement")
orchestrator.add_evidence("P3", "text", "person3.txt", "Third party statement")

# 3. Run comparative analysis
psychology = orchestrator.run_comparative_psychology()
risk = orchestrator.run_risk_assessment()

# 4. Generate investigative checklist
checklist = orchestrator.run_checklist_generation()
print(f"Next steps: {len(checklist)} items identified")

# 5. Export comprehensive report
orchestrator.export_results(results, "multi_party_analysis.json")
```

### Workflow 3: Quick Contradiction Check
```python
from src.contradiction_timeline import ContradictionTimeline

# Quick contradiction analysis
timeline = ContradictionTimeline()
timeline.add_statement("John", "I was at home", "2025-01-01 20:00")
timeline.add_statement("John", "I was at the store", "2025-01-01 21:00")

contradictions = timeline.detect_contradictions()
if contradictions:
    print(f"Found {len(contradictions)} contradictions")
    timeline.export_html("quick_contradictions.html")
```

---

## Troubleshooting

### Common Issues

**Import Error: Module not found**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

**Audio Processing Error**
```bash
# Install audio dependencies
pip install librosa soundfile pydub

# On Windows, may need ffmpeg
pip install ffmpeg-python
```

**Google AI Not Working**
```bash
# Set API key
export GEMINI_API_KEY="your_key"

# Check key validity
python -c "import google.generativeai as genai; genai.configure(api_key='your_key')"
```

**Performance Issues**
```python
# Enable caching
from src.config import CACHE_ENABLED
CACHE_ENABLED = True

# Reduce batch size
from src.config import BATCH_SIZE
BATCH_SIZE = 16
```

---

## Advanced Usage

### Custom Analysis Pipeline
```python
from src.alert_system import AlertSystem
from src.evidence_scoring import EvidenceScoringSystem
from src.comparative_psychology import ComparativePsychologyAnalyzer

# Custom pipeline
def custom_analysis(evidence_files):
    alerts = AlertSystem()
    scorer = EvidenceScoringSystem()
    psychology = ComparativePsychologyAnalyzer()
    
    results = {}
    
    for file in evidence_files:
        # Analyze each file
        text = read_file(file)
        
        # Check for alerts
        alert_result = alerts.check_all_indicators(text)
        
        # Score evidence
        score_result = scorer.calculate_evidence_score(text)
        
        # Store results
        results[file] = {
            "alerts": alert_result,
            "score": score_result
        }
    
    return results
```

### Batch Processing
```python
from src.main_orchestrator import MainOrchestrator
import glob

# Process multiple cases
orchestrator = MainOrchestrator()

for case_folder in glob.glob("cases/*/"):
    case_name = os.path.basename(case_folder.rstrip("/"))
    
    orchestrator.create_case(case_name, f"Case: {case_name}")
    
    # Add all files in case folder
    for file in glob.glob(f"{case_folder}/*"):
        orchestrator.add_evidence(
            f"EVID_{case_name}_{os.path.basename(file)}",
            "text" if file.endswith(".txt") else "audio",
            file,
            os.path.basename(file)
        )
    
    # Run analysis
    results = orchestrator.run_complete_analysis()
    orchestrator.export_results(results, f"results_{case_name}.json")
```

---

## Next Steps

1. **Start Simple**: Use the Main Orchestrator for complete analysis
2. **Explore Modules**: Try individual modules for specific tasks
3. **Web Interface**: Use Streamlit for visual analysis
4. **Customize**: Adjust thresholds and configuration
5. **Batch Process**: Handle multiple cases efficiently

---

**Need help? Check:**
- [SYSTEM_COMPLETE.md](SYSTEM_COMPLETE.md) - Full system overview
- [FINAL_OPTIMIZATION_SUMMARY.md](FINAL_OPTIMIZATION_SUMMARY.md) - Performance details
- [GITHUB_README.md](GITHUB_README.md) - Project documentation

**Ready to analyze? Start with the Main Orchestrator!**
