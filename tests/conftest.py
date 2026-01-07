"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Add the die_waarheid directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "die_waarheid"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file"""
    audio_file = temp_dir / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio data")
    return audio_file


@pytest.fixture
def mock_transcription_result():
    """Mock transcription result"""
    return {
        'success': True,
        'text': 'This is a test transcription',
        'language': 'en',
        'duration': 10.5,
        'segments': [
            {'start': 0, 'end': 10.5, 'text': 'This is a test transcription'}
        ]
    }


@pytest.fixture
def mock_forensic_result():
    """Mock forensic analysis result"""
    return {
        'success': True,
        'stress_level': 45.2,
        'pitch_volatility': 12.3,
        'silence_ratio': 0.15,
        'intensity': {'mean': 0.5, 'std': 0.1},
        'spectral_centroid': 2000.0,
        'audio_quality': 'good'
    }


@pytest.fixture
def mock_ai_analysis():
    """Mock AI analysis result"""
    return {
        'success': True,
        'text': 'This is a test message',
        'emotion': 'neutral',
        'toxicity_score': 0.1,
        'aggression_level': 'low',
        'confidence': 0.8,
        'analysis': 'The message appears to be neutral.',
        'sentiment': 'neutral'
    }


@pytest.fixture
def mock_whisper_transcriber():
    """Mock Whisper transcriber"""
    mock = Mock()
    mock.transcribe.return_value = {
        'success': True,
        'text': 'Mock transcription',
        'duration': 5.0
    }
    return mock


@pytest.fixture
def mock_forensics_engine():
    """Mock forensics engine"""
    mock = Mock()
    mock.analyze.return_value = {
        'success': True,
        'stress_level': 50.0,
        'pitch_volatility': 10.0,
        'silence_ratio': 0.2
    }
    return mock


@pytest.fixture
def mock_ai_analyzer():
    """Mock AI analyzer"""
    mock = Mock()
    mock.analyze_message.return_value = {
        'success': True,
        'emotion': 'neutral',
        'toxicity_score': 0.0,
        'aggression_level': 'low',
        'confidence': 0.9
    }
    mock.detect_gaslighting.return_value = {
        'gaslighting_detected': False,
        'gaslighting_score': 0.0
    }
    mock.detect_toxicity.return_value = {
        'toxicity_detected': False,
        'toxicity_score': 0.0
    }
    mock.detect_narcissistic_patterns.return_value = {
        'narcissistic_patterns_detected': False,
        'narcissism_score': 0.0
    }
    return mock


@pytest.fixture
def mock_speaker_system():
    """Mock speaker identification system"""
    mock = Mock()
    mock.get_all_participants.return_value = []
    mock.identify_speaker.return_value = "Unknown"
    return mock


@pytest.fixture
def sample_text_messages():
    """Sample text messages for testing"""
    return [
        "Hello, how are you?",
        "I'm doing great, thanks!",
        "That's wonderful to hear.",
        "I hate you, you're stupid!",
        "Why are you so angry?",
        "I love this product, it's amazing!"
    ]


@pytest.fixture(autouse=True)
def disable_api_calls(monkeypatch):
    """Disable actual API calls during tests"""
    # Mock environment variables
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test_token")
    
    # Mock API calls
    import sys
    from unittest.mock import Mock
    
    # Mock google.generativeai
    mock_genai = Mock()
    mock_genai.configure = Mock()
    mock_genai.GenerativeModel = Mock()
    sys.modules['google.generativeai'] = mock_genai
    
    # Mock librosa
    mock_librosa = Mock()
    mock_librosa.load = Mock(return_value=(Mock(), 22050))
    mock_librosa.pyin = Mock(return_value=(Mock(), Mock(), Mock()))
    sys.modules['librosa'] = mock_librosa
