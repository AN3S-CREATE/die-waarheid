"""
Integration tests for the complete pipeline
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline_processor import PipelineProcessor


class TestPipelineIntegration:
    """Integration tests for the complete analysis pipeline"""

    @patch('src.pipeline_processor.WhisperTranscriber')
    @patch('src.pipeline_processor.ForensicsEngine')
    @patch('src.pipeline_processor.AIAnalyzer')
    @patch('src.pipeline_processor.SpeakerIdentificationSystem')
    def test_full_pipeline_integration(self, mock_speaker, mock_ai, mock_forensics, mock_whisper):
        """Test full pipeline integration with all components"""
        # Setup all mocks
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = {
            'success': True,
            'text': 'This is a test message for analysis',
            'duration': 15.0
        }
        mock_whisper.return_value = mock_whisper_instance
        
        mock_forensics_instance = Mock()
        mock_forensics_instance.analyze.return_value = {
            'success': True,
            'stress_level': 55.0,
            'pitch_volatility': 25.0,
            'silence_ratio': 0.25,
            'intensity': {'mean': 0.6, 'std': 0.15, 'max': 0.9},
            'spectral_centroid': 2500.0,
            'audio_quality': 'good'
        }
        mock_forensics.return_value = mock_forensics_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.analyze_message.return_value = {
            'success': True,
            'analysis': 'The message shows moderate stress levels',
            'sentiment': 'neutral',
            'emotion': 'neutral',
            'toxicity_score': 0.2,
            'aggression_level': 'medium',
            'confidence': 0.85
        }
        mock_ai_instance.detect_gaslighting.return_value = {
            'gaslighting_detected': False,
            'gaslighting_score': 0.1
        }
        mock_ai_instance.detect_toxicity.return_value = {
            'toxicity_detected': False,
            'toxicity_score': 0.2
        }
        mock_ai_instance.detect_narcissistic_patterns.return_value = {
            'narcissistic_patterns_detected': False,
            'narcissism_score': 0.0
        }
        mock_ai.return_value = mock_ai_instance
        
        mock_speaker_instance = Mock()
        mock_speaker_instance.get_all_participants.return_value = ['speaker_1', 'speaker_2']
        mock_speaker_instance.identify_speaker.return_value = 'speaker_1'
        mock_speaker.return_value = mock_speaker_instance
        
        # Create processor and run pipeline
        processor = PipelineProcessor("INTEGRATION_TEST")
        processor.transcriber = mock_whisper_instance
        
        # Process a test file
        test_file = Path("test_integration.mp3")
        test_file.touch()  # Create empty file
        
        result = processor.process_voice_note(test_file)
        
        # Verify all components were called
        mock_whisper_instance.transcribe.assert_called_once()
        mock_forensics_instance.analyze.assert_called_once()
        mock_ai_instance.analyze_message.assert_called_once()
        mock_speaker_instance.identify_speaker.assert_called_once()
        
        # Verify result structure
        assert result['success']
        assert result['filename'] == 'test_integration.mp3'
        assert result['transcription'] == 'This is a test message for analysis'
        assert result['ai_interpretation'] == 'The message shows moderate stress levels'
        assert result['stress_level'] == 55.0
        assert result['risk_score'] > 0
        assert result['identified_speaker'] == 'speaker_1'
        assert len(result['deception_indicators']) >= 0
        
        # Clean up
        test_file.unlink()

    @patch('src.pipeline_processor.WhisperTranscriber')
    @patch('src.pipeline_processor.ForensicsEngine')
    @patch('src.pipeline_processor.AIAnalyzer')
    def test_pipeline_with_ai_failure(self, mock_ai, mock_forensics, mock_whisper):
        """Test pipeline behavior when AI analysis fails"""
        # Setup mocks
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = {
            'success': True,
            'text': 'Test message',
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
        
        # AI analysis fails
        mock_ai_instance = Mock()
        mock_ai_instance.analyze_message.return_value = {
            'success': False,
            'error': 'API quota exceeded',
            'error_type': 'quota_exceeded'
        }
        mock_ai_instance.detect_gaslighting.return_value = {'gaslighting_detected': False}
        mock_ai_instance.detect_toxicity.return_value = {'toxicity_detected': False}
        mock_ai_instance.detect_narcissistic_patterns.return_value = {'narcissistic_patterns_detected': False}
        mock_ai.return_value = mock_ai_instance
        
        # Process with AI failure
        processor = PipelineProcessor()
        processor.transcriber = mock_whisper_instance
        
        test_file = Path("test_ai_fail.mp3")
        test_file.touch()
        
        result = processor.process_voice_note(test_file)
        
        # Should still succeed with fallback
        assert result['success']
        assert "quota exceeded" in result['ai_interpretation']
        assert len(result['errors']) > 0
        
        test_file.unlink()

    def test_batch_processing_integration(self):
        """Test batch processing with multiple files"""
        with patch('src.pipeline_processor.WhisperTranscriber') as mock_whisper, \
             patch('src.pipeline_processor.ForensicsEngine') as mock_forensics, \
             patch('src.pipeline_processor.AIAnalyzer') as mock_ai, \
             patch('src.pipeline_processor.SpeakerIdentificationSystem') as mock_speaker:
            
            # Setup mocks
            mock_whisper.return_value.transcribe.return_value = {
                'success': True,
                'text': 'Batch test message',
                'duration': 10.0
            }
            
            mock_forensics.return_value.analyze.return_value = {
                'success': True,
                'stress_level': 40.0,
                'pitch_volatility': 20.0,
                'silence_ratio': 0.2
            }
            
            mock_ai.return_value.analyze_message.return_value = {
                'success': True,
                'analysis': 'Analysis result',
                'sentiment': 'neutral'
            }
            mock_ai.return_value.detect_gaslighting.return_value = {'gaslighting_detected': False}
            mock_ai.return_value.detect_toxicity.return_value = {'toxicity_detected': False}
            mock_ai.return_value.detect_narcissistic_patterns.return_value = {'narcissistic_patterns_detected': False}
            
            mock_speaker.return_value.get_all_participants.return_value = []
            
            # Create test files
            test_files = []
            for i in range(3):
                file_path = Path(f"test_batch_{i}.mp3")
                file_path.touch()
                test_files.append(file_path)
            
            # Process batch
            processor = PipelineProcessor()
            processor.transcriber = mock_whisper.return_value
            
            results = processor.process_batch(test_files, max_workers=2)
            
            # Verify results
            assert len(results) == 3
            for result in results:
                assert result['success']
                assert result['transcription'] == 'Batch test message'
            
            # Clean up
            for file_path in test_files:
                file_path.unlink()

    def test_error_propagation_integration(self):
        """Test error handling and propagation through the pipeline"""
        with patch('src.pipeline_processor.WhisperTranscriber') as mock_whisper:
            # Transcription fails
            mock_whisper.return_value.transcribe.return_value = {
                'success': False,
                'error': 'Audio format not supported'
            }
            
            processor = PipelineProcessor()
            processor.transcriber = mock_whisper.return_value
            
            test_file = Path("test_error.mp3")
            test_file.touch()
            
            result = processor.process_voice_note(test_file)
            
            # Should handle error gracefully
            assert result['success']  # Overall success despite errors
            assert len(result['errors']) > 0
            assert "Audio format not supported" in result['errors'][0]
            assert result['transcription'] == ''
            
            test_file.unlink()

    def test_data_flow_integrity(self):
        """Test that data flows correctly between components"""
        with patch('src.pipeline_processor.WhisperTranscriber') as mock_whisper, \
             patch('src.pipeline_processor.ForensicsEngine') as mock_forensics, \
             patch('src.pipeline_processor.AIAnalyzer') as mock_ai:
            
            # Setup specific test data
            test_transcription = "I am feeling very stressed and angry about this situation!"
            
            mock_whisper.return_value.transcribe.return_value = {
                'success': True,
                'text': test_transcription,
                'duration': 12.0
            }
            
            mock_forensics.return_value.analyze.return_value = {
                'success': True,
                'stress_level': 75.0,  # High stress
                'pitch_volatility': 40.0,  # High volatility
                'silence_ratio': 0.35  # High silence
            }
            
            mock_ai.return_value.analyze_message.return_value = {
                'success': True,
                'analysis': 'High stress and anger detected',
                'sentiment': 'negative',
                'emotion': 'angry'
            }
            mock_ai.return_value.detect_gaslighting.return_value = {'gaslighting_detected': False}
            mock_ai.return_value.detect_toxicity.return_value = {'toxicity_detected': True}
            mock_ai.return_value.detect_narcissistic_patterns.return_value = {'narcissistic_patterns_detected': False}
            
            processor = PipelineProcessor()
            processor.transcriber = mock_whisper.return_value
            
            test_file = Path("test_flow.mp3")
            test_file.touch()
            
            result = processor.process_voice_note(test_file)
            
            # Verify data flow
            assert result['transcription'] == test_transcription
            assert result['stress_level'] == 75.0
            assert result['ai_interpretation'] == 'High stress and anger detected'
            assert result['risk_score'] > 70  # Should be high risk
            assert len(result['deception_indicators']) > 0  # Should detect deception
            assert result['toxicity']['toxicity_detected'] is True
            
            test_file.unlink()
