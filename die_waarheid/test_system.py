"""
Die Waarheid - System Test Script
Run this to verify all components are working correctly.
"""

import sys
import os

# Setup path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("DIE WAARHEID - SYSTEM TEST")
    print("=" * 60)
    
    results = {
        'passed': [],
        'failed': []
    }
    
    # Critical dependencies
    print("\n[1] CRITICAL DEPENDENCIES")
    print("-" * 40)
    
    critical = [
        ('streamlit', 'Streamlit (Dashboard)'),
        ('whisper', 'OpenAI Whisper (Transcription)'),
        ('google.generativeai', 'Google Generative AI'),
        ('librosa', 'Librosa (Audio Processing)'),
        ('pandas', 'Pandas (Data Processing)'),
        ('numpy', 'NumPy'),
        ('sqlalchemy', 'SQLAlchemy (Database)'),
    ]
    
    for module, name in critical:
        try:
            __import__(module)
            print(f"    ✓ {name}")
            results['passed'].append(name)
        except ImportError as e:
            print(f"    ✗ {name}: {e}")
            results['failed'].append((name, str(e)))
    
    # Core modules
    print("\n[2] CORE ANALYSIS MODULES")
    print("-" * 40)
    
    modules = [
        ('main_orchestrator', 'MainOrchestrator', 'Main Orchestrator'),
        ('whisper_transcriber', 'WhisperTranscriber', 'Whisper Transcriber'),
        ('afrikaans_audio', 'AudioLayerSeparator', 'Audio Layer Separator'),
        ('afrikaans_processor', 'AfrikaansProcessor', 'Afrikaans Processor'),
        ('diarization', 'DiarizationPipeline', 'Speaker Diarization'),
        ('forensics', 'ForensicsEngine', 'Audio Forensics'),
        ('text_forensics', 'TextForensicsEngine', 'Text Forensics'),
        ('multilingual_support', 'MultilingualAnalyzer', 'Multilingual Analyzer'),
    ]
    
    for module, classname, name in modules:
        try:
            mod = __import__(module)
            cls = getattr(mod, classname)
            print(f"    ✓ {name}")
            results['passed'].append(name)
        except Exception as e:
            print(f"    ✗ {name}: {str(e)[:50]}")
            results['failed'].append((name, str(e)))
    
    # 8 Recommended modules
    print("\n[3] 8 RECOMMENDED MODULES")
    print("-" * 40)
    
    recommended = [
        ('alert_system', 'AlertSystem', 'Alert System'),
        ('evidence_scoring', 'EvidenceScoringSystem', 'Evidence Scoring'),
        ('investigative_checklist', 'InvestigativeChecklistGenerator', 'Investigative Checklist'),
        ('contradiction_timeline', 'ContradictionTimelineAnalyzer', 'Contradiction Timeline'),
        ('narrative_reconstruction', 'NarrativeReconstructor', 'Narrative Reconstruction'),
        ('comparative_psychology', 'ComparativePsychologyAnalyzer', 'Comparative Psychology'),
        ('risk_escalation_matrix', 'RiskEscalationMatrix', 'Risk Escalation Matrix'),
        ('multilingual_support', 'MultilingualAnalyzer', 'Multilingual Support'),
    ]
    
    for module, classname, name in recommended:
        try:
            mod = __import__(module)
            cls = getattr(mod, classname)
            print(f"    ✓ {name}")
            results['passed'].append(name)
        except Exception as e:
            print(f"    ✗ {name}: {str(e)[:50]}")
            results['failed'].append((name, str(e)))
    
    # Config
    print("\n[4] CONFIGURATION")
    print("-" * 40)
    
    config_items = [
        'WHISPER_MODEL_SIZE',
        'TARGET_SAMPLE_RATE',
        'GEMINI_API_KEY',
        'GASLIGHTING_THRESHOLD',
        'TOXICITY_THRESHOLD',
        'STRESS_WEIGHTS',
    ]
    
    try:
        import config
        for item in config_items:
            if hasattr(config, item):
                print(f"    ✓ {item}")
                results['passed'].append(f"Config: {item}")
            else:
                print(f"    ✗ {item}: Not found")
                results['failed'].append((f"Config: {item}", "Not found"))
    except Exception as e:
        print(f"    ✗ Config import failed: {e}")
        results['failed'].append(("Config", str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results['passed']) + len(results['failed'])
    passed = len(results['passed'])
    
    print(f"\nPassed: {passed}/{total}")
    
    if results['failed']:
        print(f"\n❌ {len(results['failed'])} FAILURES:")
        for name, error in results['failed']:
            print(f"   - {name}")
    else:
        print("\n✅ ALL TESTS PASSED - SYSTEM READY")
    
    return len(results['failed']) == 0


def test_dashboard():
    """Test dashboard can be imported."""
    print("\n[5] DASHBOARD")
    print("-" * 40)
    
    try:
        # Just check syntax
        import ast
        with open('dashboard_complete.py', 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("    ✓ dashboard_complete.py syntax OK")
        return True
    except Exception as e:
        print(f"    ✗ Dashboard error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    test_dashboard()
    
    print("\n" + "=" * 60)
    if success:
        print("🟢 SYSTEM READY - Run: streamlit run dashboard_complete.py")
    else:
        print("🔴 FIX FAILURES ABOVE BEFORE RUNNING")
    print("=" * 60)
