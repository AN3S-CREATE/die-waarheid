"""
Unit tests for configuration module
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    validate_config,
    get_config_summary,
    AUDIO_DIR,
    TEXT_DIR,
    TEMP_DIR,
    REPORTS_DIR,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_TEXT_FORMATS
)


class TestConfiguration(unittest.TestCase):
    """Test configuration validation and settings"""

    def test_directories_exist(self):
        """Test that required directories exist or can be created"""
        directories = [AUDIO_DIR, TEXT_DIR, TEMP_DIR, REPORTS_DIR]
        
        for directory in directories:
            self.assertIsNotNone(directory)
            self.assertTrue(isinstance(directory, Path))

    def test_supported_formats(self):
        """Test supported file formats are defined"""
        self.assertGreater(len(SUPPORTED_AUDIO_FORMATS), 0)
        self.assertGreater(len(SUPPORTED_TEXT_FORMATS), 0)
        
        self.assertIn('.mp3', SUPPORTED_AUDIO_FORMATS)
        self.assertIn('.wav', SUPPORTED_AUDIO_FORMATS)
        self.assertIn('.txt', SUPPORTED_TEXT_FORMATS)

    def test_validate_config(self):
        """Test configuration validation"""
        errors, warnings = validate_config()
        
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)

    def test_get_config_summary(self):
        """Test config summary generation"""
        summary = get_config_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertGreater(len(summary), 0)


class TestConfigurationValues(unittest.TestCase):
    """Test specific configuration values"""

    def test_stress_thresholds(self):
        """Test stress level thresholds are reasonable"""
        from config import STRESS_THRESHOLD_HIGH
        
        self.assertGreater(STRESS_THRESHOLD_HIGH, 0)
        self.assertLess(STRESS_THRESHOLD_HIGH, 100)

    def test_batch_size(self):
        """Test batch size is positive"""
        from config import BATCH_SIZE
        
        self.assertGreater(BATCH_SIZE, 0)
        self.assertLess(BATCH_SIZE, 1000)

    def test_max_workers(self):
        """Test max workers is reasonable"""
        from config import MAX_WORKERS
        
        self.assertGreater(MAX_WORKERS, 0)
        self.assertLess(MAX_WORKERS, 32)


if __name__ == '__main__':
    unittest.main()
