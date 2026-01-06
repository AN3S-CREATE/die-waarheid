"""
Unit tests for ForensicsEngine
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.forensics import ForensicsEngine


class TestForensicsEngine:
    """Test cases for ForensicsEngine class"""

    def test_init(self):
        """Test ForensicsEngine initialization"""
        engine = ForensicsEngine(sample_rate=16000, use_cache=False)
        assert engine.sample_rate == 16000
        assert engine.audio_data is None
        assert engine.filename is None
        assert engine.cache is None

    @patch('src.forensics.librosa')
    def test_load_audio_success(self, mock_librosa):
        """Test successful audio loading"""
        # Setup mock
        mock_librosa.load.return_value = (np.array([1, 2, 3, 4, 5]), 22050)
        
        engine = ForensicsEngine()
        success, message = engine.load_audio(Path("test.mp3"))
        
        assert success
        assert "Successfully loaded" in message
        assert engine.filename == "test.mp3"
        assert len(engine.audio_data) == 5

    def test_load_audio_file_not_found(self):
        """Test loading non-existent file"""
        engine = ForensicsEngine()
        success, message = engine.load_audio(Path("nonexistent.mp3"))
        
        assert not success
        assert "File not found" in message

    @patch('src.forensics.librosa')
    def test_load_audio_error(self, mock_librosa):
        """Test handling of loading error"""
        mock_librosa.load.side_effect = Exception("Audio error")
        
        engine = ForensicsEngine()
        success, message = engine.load_audio(Path("test.mp3"))
        
        assert not success
        assert "Error loading audio" in message

    @patch('src.forensics.librosa')
    def test_extract_pitch_features(self, mock_librosa):
        """Test pitch feature extraction"""
        # Setup mock data
        mock_audio = np.array([1, 2, 3, 4] * 1000)  # 4000 samples
        mock_librosa.pyin.return_value = (
            np.array([100, 110, 105, 95]),  # pitch
            np.array([True, True, True, True]),  # voiced
            np.array([0.0, 0.5, 1.0, 1.5])  # times
        )
        
        engine = ForensicsEngine()
        engine.audio_data = mock_audio
        
        mean_pitch, pitch_std = engine._extract_pitch_features()
        
        assert isinstance(mean_pitch, float)
        assert isinstance(pitch_std, float)
        assert mean_pitch > 0

    @patch('src.forensics.librosa')
    def test_extract_pitch_features_no_voiced(self, mock_librosa):
        """Test pitch extraction with no voiced frames"""
        mock_audio = np.array([1, 2, 3, 4] * 1000)
        mock_librosa.pyin.return_value = (
            np.array([np.nan, np.nan, np.nan]),  # no pitch
            np.array([False, False, False]),  # not voiced
            np.array([0.0, 0.5, 1.0])
        )
        
        engine = ForensicsEngine()
        engine.audio_data = mock_audio
        
        mean_pitch, pitch_std = engine._extract_pitch_features()
        
        assert mean_pitch == 0.0
        assert pitch_std == 0.0

    def test_extract_silence_ratio(self):
        """Test silence ratio extraction"""
        # Test with normal audio
        engine = ForensicsEngine()
        engine.audio_data = np.array([0.1, 0.2, 0.0, 0.0, 0.3, 0.4])
        
        silence_ratio = engine._extract_silence_ratio()
        
        assert 0 <= silence_ratio <= 1
        assert silence_ratio == 2/6  # 2 silent frames out of 6

    def test_extract_intensity_features(self):
        """Test intensity feature extraction"""
        engine = ForensicsEngine()
        engine.audio_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        intensity = engine._extract_intensity_features()
        
        assert 'mean' in intensity
        assert 'std' in intensity
        assert 'max' in intensity
        assert intensity['mean'] > 0

    def test_extract_spectral_features(self):
        """Test spectral feature extraction"""
        engine = ForensicsEngine()
        engine.audio_data = np.array([1, 2, 3, 4, 5] * 1000)
        
        spectral_centroid = engine._extract_spectral_features()
        
        assert isinstance(spectral_centroid, float)
        assert spectral_centroid > 0

    @patch.object(ForensicsEngine, '_extract_pitch_features')
    @patch.object(ForensicsEngine, '_extract_silence_ratio')
    @patch.object(ForensicsEngine, '_extract_intensity_features')
    @patch.object(ForensicsEngine, '_extract_spectral_features')
    def test_analyze_success(self, mock_spectral, mock_intensity, mock_silence, mock_pitch):
        """Test successful analysis"""
        # Setup mocks
        mock_pitch.return_value = (150.0, 20.0)
        mock_silence.return_value = 0.15
        mock_intensity.return_value = {'mean': 0.5, 'std': 0.1, 'max': 0.8}
        mock_spectral.return_value = 2000.0
        
        engine = ForensicsEngine()
        engine.filename = "test.mp3"
        
        result = engine.analyze(Path("test.mp3"))
        
        assert result['success']
        assert result['stress_level'] == pytest.approx(45.0, rel=1e-2)
        assert result['pitch_volatility'] == pytest.approx(20.0, rel=1e-2)
        assert result['silence_ratio'] == 0.15
        assert result['intensity'] == {'mean': 0.5, 'std': 0.1, 'max': 0.8}
        assert result['spectral_centroid'] == 2000.0

    def test_analyze_no_audio_loaded(self):
        """Test analysis without loaded audio"""
        engine = ForensicsEngine()
        
        result = engine.analyze(Path("test.mp3"))
        
        assert result['success']
        assert result['stress_level'] == 0.0
        assert result['pitch_volatility'] == 0.0

    def test_calculate_stress_level(self):
        """Test stress level calculation"""
        engine = ForensicsEngine()
        
        # High stress scenario
        stress = engine._calculate_stress_level(
            mean_pitch=200.0,
            pitch_std=50.0,
            silence_ratio=0.4
        )
        assert stress > 60
        
        # Low stress scenario
        stress = engine._calculate_stress_level(
            mean_pitch=100.0,
            pitch_std=10.0,
            silence_ratio=0.1
        )
        assert stress < 40

    def test_calculate_pitch_volatility(self):
        """Test pitch volatility calculation"""
        engine = ForensicsEngine()
        
        # High volatility
        volatility = engine._calculate_pitch_volatility(50.0)
        assert volatility == 50.0
        
        # Zero volatility
        volatility = engine._calculate_pitch_volatility(0.0)
        assert volatility == 0.0

    def test_assess_audio_quality(self):
        """Test audio quality assessment"""
        engine = ForensicsEngine()
        
        # Good quality
        quality = engine._assess_audio_quality(
            duration=60.0,
            signal_to_noise=30.0,
            zero_crossing_rate=0.1
        )
        assert quality == 'good'
        
        # Poor quality
        quality = engine._assess_audio_quality(
            duration=5.0,
            signal_to_noise=5.0,
            zero_crossing_rate=0.5
        )
        assert quality == 'poor'

    def test_get_summary_statistics(self):
        """Test summary statistics"""
        engine = ForensicsEngine()
        
        stats = engine.get_summary_statistics()
        
        assert 'total_analyzed' in stats
        assert 'avg_stress_level' in stats
        assert 'avg_pitch_volatility' in stats
        assert 'avg_silence_ratio' in stats

    @patch('src.forensics.librosa')
    def test_analyze_with_cache(self, mock_librosa):
        """Test analysis with caching enabled"""
        mock_librosa.load.return_value = (np.array([1, 2, 3, 4]), 22050)
        
        engine = ForensicsEngine(use_cache=True)
        
        # First analysis
        result1 = engine.analyze(Path("test.mp3"))
        
        # Second analysis should use cache
        result2 = engine.analyze(Path("test.mp3"))
        
        assert result1['success']
        assert result2['success']
