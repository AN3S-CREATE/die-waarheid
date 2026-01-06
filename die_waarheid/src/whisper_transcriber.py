"""
Whisper Transcription Engine for Die Waarheid
Handles audio transcription with Afrikaans language support
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import whisper
import torch

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_OPTIONS,
    TEMP_DIR
)

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Whisper-based transcription engine with model caching
    Supports multiple languages with Afrikaans optimization
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
    
    # Class-level model cache to prevent reloading
    _model_cache = {}
    _cache_lock = threading.Lock()

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        """
        Initialize Whisper transcriber with model caching

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size not in self.AVAILABLE_MODELS:
            logger.warning(f"Invalid model size '{model_size}', using 'medium'")
            model_size = "medium"

        self.model_size = model_size
        self.model = None
        self.device = self._get_optimal_device()
        self.load_model()

    @staticmethod
    def _get_optimal_device() -> str:
        """
        Determine optimal device for model inference
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        # Check environment variable first
        device_env = os.getenv('DEVICE', 'auto').lower()
        if device_env != 'auto':
            return device_env
        
        # Auto-detect optimal device
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logger.info("MPS available: Using Apple Silicon GPU")
        else:
            device = 'cpu'
            logger.info("Using CPU for inference")
        
        return device

    def load_model(self) -> bool:
        """
        Load Whisper model with caching to prevent reloading
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        cache_key = f"{self.model_size}_{self.device}"
        
        # Check if model is already cached
        with self._cache_lock:
            if cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                logger.info(f"Using cached Whisper {self.model_size} model on {self.device}")
                return True
        
        try:
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            
            # Load model with device specification
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            # Cache the model for future use
            with self._cache_lock:
                self._model_cache[cache_key] = self.model
                logger.info(f"Cached Whisper {self.model_size} model for reuse")
            
            logger.info(f"Successfully loaded Whisper {self.model_size} model on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            return False

    @classmethod
    def clear_model_cache(cls):
        """Clear all cached models to free memory"""
        with cls._cache_lock:
            cls._model_cache.clear()
            logger.info("Cleared Whisper model cache")

    @classmethod
    def get_cache_info(cls) -> Dict:
        """Get information about cached models"""
        with cls._cache_lock:
            return {
                "cached_models": list(cls._model_cache.keys()),
                "cache_size": len(cls._model_cache)
            }

    def transcribe(
        self,
        audio_file: Path,
        language: str = "af",
        verbose: bool = False
    ) -> Dict:
        """
        Transcribe audio file using Whisper

        Args:
            audio_file: Path to audio file
            language: Language code (default: 'af' for Afrikaans)
            verbose: Print progress information

        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {
                'success': False,
                'message': 'Model not loaded',
                'filename': str(audio_file)
            }

        try:
            audio_file = Path(audio_file)

            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return {
                    'success': False,
                    'message': f'File not found: {audio_file}',
                    'filename': str(audio_file)
                }

            logger.info(f"Transcribing: {audio_file.name}")

            options = WHISPER_OPTIONS.copy()
            options['language'] = language
            options['verbose'] = verbose

            result = self.model.transcribe(str(audio_file), **options)

            segments = result.get('segments', [])
            text = result.get('text', '')

            logger.info(f"Transcription complete: {len(segments)} segments, {len(text)} characters")

            return {
                'success': True,
                'filename': audio_file.name,
                'language': language,
                'text': text,
                'segments': segments,
                'duration': result.get('duration', 0),
                'language_detected': result.get('language', language)
            }

        except Exception as e:
            logger.error(f"Error transcribing {audio_file.name}: {str(e)}")
            return {
                'success': False,
                'message': f'Transcription error: {str(e)}',
                'filename': audio_file.name
            }

    def transcribe_with_timestamps(
        self,
        audio_file: Path,
        language: str = "af"
    ) -> Dict:
        """
        Transcribe with detailed timestamp information

        Args:
            audio_file: Path to audio file
            language: Language code

        Returns:
            Dictionary with timestamped segments
        """
        result = self.transcribe(audio_file, language)

        if not result.get('success'):
            return result

        segments = result.get('segments', [])
        timestamped_segments = []

        for segment in segments:
            timestamped_segments.append({
                'id': segment.get('id'),
                'start': segment.get('start'),
                'end': segment.get('end'),
                'text': segment.get('text'),
                'confidence': segment.get('confidence', 0.0)
            })

        result['timestamped_segments'] = timestamped_segments
        logger.debug(f"Created {len(timestamped_segments)} timestamped segments")

        return result

    def batch_transcribe(
        self,
        audio_files: List[Path],
        language: str = "af"
    ) -> List[Dict]:
        """
        Transcribe multiple audio files

        Args:
            audio_files: List of audio file paths
            language: Language code

        Returns:
            List of transcription results
        """
        results = []

        for idx, audio_file in enumerate(audio_files):
            logger.info(f"Transcribing {idx + 1}/{len(audio_files)}: {audio_file.name}")
            result = self.transcribe(audio_file, language)
            results.append(result)

        logger.info(f"Batch transcription complete: {len(results)} files processed")
        return results

    def get_model_info(self) -> Dict:
        """
        Get information about loaded model

        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'model_loaded': self.model is not None,
            'available_models': self.AVAILABLE_MODELS,
            'default_language': 'af',
            'supported_languages': [
                'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo',
                'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es',
                'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw',
                'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv',
                'ka', 'kk', 'km', 'kn', 'ko', 'ky', 'la', 'lb', 'ln', 'lo',
                'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne',
                'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
                'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su',
                'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt',
                'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
            ]
        }

    def change_model(self, model_size: str) -> bool:
        """
        Change to a different model size

        Args:
            model_size: New model size

        Returns:
            True if model changed successfully
        """
        if model_size not in self.AVAILABLE_MODELS:
            logger.error(f"Invalid model size: {model_size}")
            return False

        try:
            logger.info(f"Switching from {self.model_size} to {model_size} model")
            self.model_size = model_size
            self.load_model()
            return True

        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            return False


class TranscriptionPipeline:
    """
    Complete transcription pipeline combining forensics and transcription
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.transcriber = WhisperTranscriber(model_size)

    def transcribe_audio_file(
        self,
        audio_file: Path,
        language: str = "af",
        with_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe a single audio file

        Args:
            audio_file: Path to audio file
            language: Language code
            with_timestamps: Include timestamp information

        Returns:
            Transcription result dictionary
        """
        if with_timestamps:
            return self.transcriber.transcribe_with_timestamps(audio_file, language)
        else:
            return self.transcriber.transcribe(audio_file, language)

    def process_batch(
        self,
        audio_files: List[Path],
        language: str = "af"
    ) -> Dict:
        """
        Process batch of audio files

        Args:
            audio_files: List of audio file paths
            language: Language code

        Returns:
            Dictionary with batch processing results
        """
        logger.info(f"Starting batch processing of {len(audio_files)} files")

        results = self.transcriber.batch_transcribe(audio_files, language)

        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {
            'total_files': len(audio_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }


if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    print("Whisper Model Info:")
    import json
    print(json.dumps(transcriber.get_model_info(), indent=2))
