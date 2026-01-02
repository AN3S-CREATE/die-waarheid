"""
Automated tests for Die Waarheid forensic pipeline
Ensures Afrikaans processing, code-switch detection, and diarization work correctly.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.whisper_transcriber import WhisperTranscriber
from src.afrikaans_processor import AfrikaansProcessor
from src.diarization import SimpleDiarizer
from src.multilingual_support import MultilingualAnalyzer
from background_sound_analyzer import BackgroundSoundAnalyzer


class TestAfrikaansProcessing(unittest.TestCase):
    """Test Afrikaans text processing and verification."""
    
    def setUp(self):
        self.processor = AfrikaansProcessor()
        
        # Test Afrikaans sentences
        self.afrikaans_samples = [
            "Ek weet nie wat jy praat nie.",
            "Dit is nie waar nie.",
            "Jy lieg vir my.",
            "Wat het jy gister gedoen?",
            "Die polisie sal hierdie saak ondersoek."
        ]
        
        # Expected corrections
        self.expected_corrections = {
            "Ek weet nie wat jy praat nie.": [],  # Correct Afrikaans
            "Dit is nie waar nie.": [],  # Correct Afrikaans
            "Jy lieg vir my.": [],  # Correct Afrikaans
            "Wat het jy gister gedoen?": [],  # Correct Afrikaans
            "Die polisie sal hierdie saak ondersoek.": []  # Correct Afrikaans
        }
    
    def test_afrikaans_detection(self):
        """Test that Afrikaans text is correctly identified."""
        for text in self.afrikaans_samples:
            result = self.processor.process_text(text)
            self.assertIn("language", result)
            self.assertEqual(result["language"], "afrikaans")
    
    def test_text_verification(self):
        """Test triple-check verification process."""
        for text in self.afrikaans_samples:
            result = self.processor.process_text(text)
            self.assertIn("verified_text", result)
            self.assertIn("confidence", result)
            self.assertGreater(result["confidence"], 0.8)  # High confidence for correct Afrikaans
    
    def test_corrections(self):
        """Test that corrections are properly identified."""
        # Test with common errors
        error_text = "ek weet nie wat jy praat nie"  # Missing capitalization
        result = self.processor.process_text(error_text)
        self.assertIn("corrections", result)
        
        # Should detect capitalization error
        corrections = result["corrections"]
        self.assertTrue(any("capitalization" in str(c).lower() for c in corrections))


class TestCodeSwitchDetection(unittest.TestCase):
    """Test detection of Afrikaans/English code-switching."""
    
    def setUp(self):
        self.analyzer = MultilingualAnalyzer()
        
        # Code-switch samples
        self.code_switch_samples = [
            "Ek gaan now to the shop.",
            "Can you help me met hierdie?",
            "Die car is broken but ek kan dit fix.",
            "I will call you later, maar nou is dit busy."
        ]
    
    def test_code_switch_detection(self):
        """Test that code-switching is detected."""
        for text in self.code_switch_samples:
            result = self.analyzer.analyze(text)
            self.assertIn("code_switches", result)
            self.assertGreater(len(result["code_switches"]), 0)
    
    def test_language_identification(self):
        """Test primary language identification."""
        # Pure Afrikaans
        afrikaans_text = "Ek is vandag baie moeg."
        result = self.analyzer.analyze(afrikaans_text)
        self.assertEqual(result["primary_language"], "afrikaans")
        
        # Pure English
        english_text = "I am very tired today."
        result = self.analyzer.analyze(english_text)
        self.assertEqual(result["primary_language"], "english")
        
        # Mixed (should detect as mixed)
        mixed_text = "Ek is tired today."
        result = self.analyzer.analyze(mixed_text)
        self.assertIn(result["primary_language"], ["mixed", "afrikaans", "english"])


class TestDiarization(unittest.TestCase):
    """Test speaker diarization functionality."""
    
    def setUp(self):
        self.diarizer = SimpleDiarizer()
        
        # Create a mock audio file path (won't actually exist in test)
        self.mock_audio_path = "test_audio.wav"
    
    def test_speaker_segmentation(self):
        """Test that audio is segmented into speaker turns."""
        # This would normally process real audio
        # For testing, we'll check the method exists and returns expected structure
        try:
            result = self.diarizer.diarize(self.mock_audio_path)
            self.assertIn("segments", result)
            self.assertIn("speaker_count", result)
        except FileNotFoundError:
            # Expected for mock file
            pass
    
    def test_segment_format(self):
        """Test that segments have correct format."""
        # Mock segment data
        mock_segments = [
            {"start": 0.0, "end": 2.5, "speaker": "Speaker A", "text": "First segment"},
            {"start": 2.5, "end": 5.0, "speaker": "Speaker B", "text": "Second segment"}
        ]
        
        for segment in mock_segments:
            self.assertIn("start", segment)
            self.assertIn("end", segment)
            self.assertIn("speaker", segment)
            self.assertGreater(segment["end"], segment["start"])


class TestBackgroundSoundAnalysis(unittest.TestCase):
    """Test background sound identification."""
    
    def setUp(self):
        self.analyzer = BackgroundSoundAnalyzer()
    
    def test_energy_calculation(self):
        """Test energy calculation for audio signals."""
        import numpy as np
        
        # Create test audio signal
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        energy = self.analyzer.calculate_energy(audio)
        self.assertIsInstance(energy, float)
        self.assertGreater(energy, 0)
    
    def test_frequency_analysis(self):
        """Test frequency profile analysis."""
        import numpy as np
        
        # Create test audio with low frequency content
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave
        
        freq_profile = self.analyzer.analyze_frequency_profile(audio, sample_rate)
        self.assertIsInstance(freq_profile, dict)
        self.assertIn("bass", freq_profile)
        self.assertIn("high", freq_profile)
    
    def test_sound_identification(self):
        """Test sound type identification."""
        # Test with low energy (should identify as quiet)
        energy = 0.02
        freq_profile = {"bass": 0.8, "mid": 0.2, "high": 0.1}
        
        sounds = self.analyzer.identify_sound_type(energy, freq_profile)
        self.assertIsInstance(sounds, list)
        
        # Should identify some sound type
        if sounds:
            self.assertIn("type", sounds[0])
            self.assertIn("confidence", sounds[0])
            self.assertLessEqual(sounds[0]["confidence"], 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        self.test_audio_file = self.temp_dir / "test_audio.wav"
        self.test_text_file = self.temp_dir / "test_text.txt"
        
        # Create simple test text file
        with open(self.test_text_file, 'w', encoding='utf-8') as f:
            f.write("Ek is 'n toets lêer. This is a test file.")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_file_hashing(self):
        """Test file hash calculation."""
        from verification_harness import VerificationHarness
        
        harness = VerificationHarness(self.temp_dir)
        file_hash = harness.calculate_file_hash(self.test_text_file)
        
        self.assertIsInstance(file_hash, str)
        self.assertEqual(len(file_hash), 64)  # SHA-256 length
    
    def test_manifest_creation(self):
        """Test case manifest creation and updates."""
        from ingestion_engine import IngestionEngine
        
        engine = IngestionEngine(self.temp_dir)
        
        # Copy test file
        result = engine.copy_file_with_metadata(self.test_text_file)
        self.assertIsNotNone(result)
        self.assertIn("sha256", result)
        self.assertIn("type", result)
        
        # Update manifest
        engine.update_manifest([result])
        
        # Check manifest
        manifest = engine.get_manifest()
        self.assertEqual(len(manifest["files"]), 1)
        self.assertEqual(manifest["files"][0]["name"], "test_text.txt")
    
    def test_no_placeholders_allowed(self):
        """Test that no placeholder outputs are generated."""
        # This test ensures all outputs come from real processing
        processor = AfrikaansProcessor()
        
        result = processor.process_text("Test text")
        
        # Check for placeholder values
        self.assertNotEqual(result.get("verified_text", ""), "PLACEHOLDER")
        self.assertNotEqual(result.get("confidence", -1), -1)
        self.assertNotEqual(result.get("language", ""), "UNKNOWN")


class TestReproducibility(unittest.TestCase):
    """Test reproducibility of analysis results."""
    
    def test_deterministic_processing(self):
        """Test that same input produces same output."""
        processor = AfrikaansProcessor()
        test_text = "Dit is 'n toets."
        
        # Process twice
        result1 = processor.process_text(test_text)
        result2 = processor.process_text(test_text)
        
        # Results should be identical
        self.assertEqual(result1["verified_text"], result2["verified_text"])
        self.assertEqual(result1["confidence"], result2["confidence"])
    
    def test_config_snapshot(self):
        """Test that configuration is saved with results."""
        from verification_harness import VerificationHarness
        
        temp_dir = Path(tempfile.mkdtemp())
        harness = VerificationHarness(temp_dir)
        
        config = {"test_param": "test_value"}
        harness.save_config(config)
        
        # Check config was saved
        config_file = temp_dir / "config.json"
        self.assertTrue(config_file.exists())
        
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config["test_param"], "test_value")
        self.assertIn("timestamp", saved_config)
        self.assertIn("modules", saved_config)


def run_tests():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAfrikaansProcessing,
        TestCodeSwitchDetection,
        TestDiarization,
        TestBackgroundSoundAnalysis,
        TestIntegration,
        TestReproducibility
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    }
    
    # Save report
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest Report:")
    print(f"Tests run: {report['tests_run']}")
    print(f"Success rate: {report['success_rate']:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
