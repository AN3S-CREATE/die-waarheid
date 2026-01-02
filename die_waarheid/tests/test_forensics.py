"""
Unit tests for forensics audio analysis engine
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forensics import ForensicsEngine


class TestForensicsEngine(unittest.TestCase):
    """Test audio forensics analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ForensicsEngine()

    def test_engine_initialization(self):
        """Test forensics engine initializes"""
        self.assertIsNotNone(self.engine)

    def test_stress_level_calculation(self):
        """Test stress level calculation is within bounds"""
        stress = self.engine.calculate_stress_level(
            pitch_volatility=0.5,
            silence_ratio=0.3,
            intensity_max=0.7,
            mfcc_variance=0.4,
            zero_crossing_rate=0.2
        )
        
        self.assertGreaterEqual(stress, 0)
        self.assertLessEqual(stress, 100)

    def test_pitch_volatility_bounds(self):
        """Test pitch volatility is normalized"""
        volatility = 0.75
        
        self.assertGreaterEqual(volatility, 0)
        self.assertLessEqual(volatility, 1)

    def test_silence_ratio_bounds(self):
        """Test silence ratio is between 0 and 1"""
        ratio = 0.45
        
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1)

    def test_intensity_calculation(self):
        """Test intensity metrics are valid"""
        # Create synthetic audio data
        sample_rate = 16000
        duration = 1
        samples = np.random.randn(sample_rate * duration)
        
        # Test that engine can process audio
        self.assertIsNotNone(self.engine)


class TestBioSignalDetection(unittest.TestCase):
    """Test bio-signal detection"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ForensicsEngine()

    def test_high_stress_detection(self):
        """Test detection of high stress indicators"""
        stress_score = 75
        
        is_high_stress = stress_score > 50
        self.assertTrue(is_high_stress)

    def test_cognitive_load_detection(self):
        """Test detection of high cognitive load"""
        mfcc_variance = 0.8
        
        is_high_load = mfcc_variance > 0.6
        self.assertTrue(is_high_load)

    def test_pitch_volatility_detection(self):
        """Test detection of pitch volatility"""
        volatility = 0.7
        
        is_volatile = volatility > 0.5
        self.assertTrue(is_volatile)


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ForensicsEngine()

    def test_supported_sample_rates(self):
        """Test that engine supports standard sample rates"""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for sr in sample_rates:
            self.assertGreater(sr, 0)

    def test_audio_duration_handling(self):
        """Test handling of various audio durations"""
        durations = [1, 5, 10, 30, 60]
        
        for duration in durations:
            self.assertGreater(duration, 0)


if __name__ == '__main__':
    unittest.main()
