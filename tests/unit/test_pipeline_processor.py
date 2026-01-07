"""
Unit tests for PipelineProcessor
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline_processor import PipelineProcessor


class TestPipelineProcessor:
    """Test cases for PipelineProcessor class"""

    def test_init(self):
        """Test PipelineProcessor initialization"""
        processor = PipelineProcessor("TEST_CASE")
        assert processor.case_id == "TEST_CASE"
        assert processor.transcriber is None
        assert processor.results == []
        assert processor.forensics is not None
        assert processor.speaker_system is not None
        assert processor.ai_analyzer is not None

    def test_process_voice_note_file_not_found(self):
        """Test processing non-existent file"""
        processor = PipelineProcessor()
        result = processor.process_voice_note(Path("nonexistent.mp3"))
        
        assert not result['success']
        assert "File not found" in result['errors'][0]
        assert result['risk_score'] == 0
        assert result['stress_level'] == 0
        assert result['identified_speaker'] == 'Unknown'

    @patch('src.pipeline_processor.WhisperTranscriber')
    @patch('src.pipeline_processor.ForensicsEngine')
    @patch('src.pipeline_processor.AIAnalyzer')
    @patch('src.pipeline_processor.SpeakerIdentificationSystem')
    def test_process_voice_note_success(self, mock_speaker, mock_ai, mock_forensics, mock_whisper):
        """Test successful voice note processing"""
        # Setup mocks
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = {
            'success': True,
            'text': 'Test transcription',
            'duration': 10.0
        }
        mock_whisper.return_value = mock_whisper_instance
        
        mock_forensics_instance = Mock()
        mock_forensics_instance.analyze.return_value = {
            'success': True,
            'stress_level': 30.0,
            'pitch_volatility': 15.0,
            'silence_ratio': 0.2
        }
        mock_forensics.return_value = mock_forensics_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.analyze_message.return_value = {
            'success': True,
            'analysis': 'AI analysis result',
            'sentiment': 'neutral'
        }
        mock_ai_instance.detect_gaslighting.return_value = {'gaslighting_detected': False}
        mock_ai_instance.detect_toxicity.return_value = {'toxicity_detected': False}
        mock_ai_instance.detect_narcissistic_patterns.return_value = {'narcissistic_patterns_detected': False}
        mock_ai.return_value = mock_ai_instance
        
        mock_speaker_instance = Mock()
        mock_speaker_instance.get_all_participants.return_value = []
        mock_speaker.return_value = mock_speaker_instance
        
        # Test
        processor = PipelineProcessor()
        processor.transcriber = mock_whisper_instance
        
        result = processor.process_voice_note(Path("test.mp3"))
        
        assert result['success']
        assert result['transcription'] == 'Test transcription'
        assert result['ai_interpretation'] == 'AI analysis result'
        assert result['stress_level'] == 30.0
        assert result['risk_score'] >= 0
        assert len(result['errors']) == 0

    @patch('src.pipeline_processor.WhisperTranscriber')
    def test_process_voice_note_transcription_failure(self, mock_whisper):
        """Test handling of transcription failure"""
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = {
            'success': False,
            'error': 'Transcription failed'
        }
        mock_whisper.return_value = mock_whisper_instance
        
        processor = PipelineProcessor()
        processor.transcriber = mock_whisper_instance
        
        result = processor.process_voice_note(Path("test.mp3"))
        
        assert result['success']  # Still succeeds overall
        assert "Transcription failed" in result['errors'][0]
        assert result['transcription'] == ''

    def test_detect_deception(self):
        """Test deception detection logic"""
        processor = PipelineProcessor()
        
        # Test high stress, high volatility
        indicators = processor._detect_deception(
            stress_level=80,
            pitch_volatility=50,
            silence_ratio=0.4,
            transcription=""
        )
        assert len(indicators) > 0
        assert any("stress" in i.lower() for i in indicators)
        
        # Test normal values
        indicators = processor._detect_deception(
            stress_level=20,
            pitch_volatility=10,
            silence_ratio=0.1,
            transcription=""
        )
        assert len(indicators) == 0

    def test_calculate_risk_score(self):
        """Test risk score calculation"""
        processor = PipelineProcessor()
        
        # High risk scenario
        result = {
            'stress_level': 80,
            'pitch_volatility': 60,
            'silence_ratio': 0.5,
            'deception_indicators': ['High stress detected', 'Unusual silence patterns'],
            'gaslighting': {'gaslighting_detected': True},
            'toxicity': {'toxicity_detected': True},
            'narcissism': {'narcissistic_patterns_detected': True}
        }
        score = processor._calculate_risk_score(result)
        assert score > 70
        
        # Low risk scenario
        result = {
            'stress_level': 20,
            'pitch_volatility': 10,
            'silence_ratio': 0.1,
            'deception_indicators': [],
            'gaslighting': {'gaslighting_detected': False},
            'toxicity': {'toxicity_detected': False},
            'narcissism': {'narcissistic_patterns_detected': False}
        }
        score = processor._calculate_risk_score(result)
        assert score < 40

    def test_get_chronological_results(self):
        """Test chronological sorting of results"""
        processor = PipelineProcessor()
        
        # Add results in random order
        processor.results = [
            {'timestamp': datetime(2023, 1, 3), 'filename': 'c.mp3'},
            {'timestamp': datetime(2023, 1, 1), 'filename': 'a.mp3'},
            {'timestamp': datetime(2023, 1, 2), 'filename': 'b.mp3'}
        ]
        
        sorted_results = processor.get_chronological_results()
        
        assert sorted_results[0]['filename'] == 'a.mp3'
        assert sorted_results[1]['filename'] == 'b.mp3'
        assert sorted_results[2]['filename'] == 'c.mp3'

    def test_get_summary_stats(self):
        """Test summary statistics calculation"""
        processor = PipelineProcessor()
        
        processor.results = [
            {
                'success': True,
                'risk_score': 80,
                'stress_level': 60,
                'deception_indicators': ['High stress']
            },
            {
                'success': True,
                'risk_score': 20,
                'stress_level': 30,
                'deception_indicators': []
            },
            {
                'success': False,
                'risk_score': 0,
                'stress_level': 0,
                'deception_indicators': []
            }
        ]
        
        stats = processor.get_summary_stats()
        
        assert stats['total_files'] == 3
        assert stats['high_risk_count'] == 1
        assert stats['deception_count'] == 1
        assert stats['avg_stress_level'] == 30.0
        assert stats['avg_risk_score'] == 33.33

    @patch('src.pipeline_processor.ThreadPoolExecutor')
    def test_process_batch(self, mock_executor):
        """Test batch processing with parallel execution"""
        # Setup mock executor
        mock_future = Mock()
        mock_future.result.return_value = {
            'success': True,
            'filename': 'test.mp3'
        }
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        processor = PipelineProcessor()
        files = [Path("test1.mp3"), Path("test2.mp3")]
        
        results = processor.process_batch(files, max_workers=2)
        
        assert len(results) == 2
        mock_executor.assert_called_once_with(max_workers=2)

    def test_export_results(self, temp_dir):
        """Test results export to JSON"""
        processor = PipelineProcessor()
        processor.results = [
            {
                'filename': 'test.mp3',
                'timestamp': datetime(2023, 1, 1),
                'success': True,
                'risk_score': 50
            }
        ]
        
        output_path = temp_dir / "results.json"
        processor.export_results(output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "test.mp3" in content
        assert "50" in content
