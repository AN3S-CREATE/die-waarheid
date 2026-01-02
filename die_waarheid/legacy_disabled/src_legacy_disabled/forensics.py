import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

try:
    import librosa
except ImportError:
    librosa = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

# Global variables for models
_whisper_model = None
_diarization_pipeline = None

def _load_whisper_model():
    """Load Whisper model lazily"""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            model_name = os.getenv("WHISPER_MODEL", "base")
            logger.info(f"Loading Whisper model: {model_name}")
            _whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            _whisper_model = None
    return _whisper_model

def _load_diarization_pipeline():
    """Load diarization pipeline lazily"""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        try:
            from pyannote.audio import Pipeline
            import torch
            
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                logger.info("Loading speaker diarization model...")
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if torch.cuda.is_available():
                    _diarization_pipeline.to(torch.device("cuda"))
                logger.info("Speaker diarization model loaded successfully")
            else:
                logger.warning("No HuggingFace token provided")
        except Exception as e:
            logger.warning(f"Failed to load diarization model: {e}")
            _diarization_pipeline = None
    return _diarization_pipeline

class AudioAnalyzer:
    """Handles the analysis of audio files for forensic purposes."""
    
    def __init__(self):
        # Don't load models in __init__ - load them lazily when needed
        pass
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Perform comprehensive audio analysis including stress indicators and speaker diarization.
        
        Args:
            audio_path: Path to the audio file to analyze
        
        Returns:
            Dictionary containing analysis results
        """
        result = {
            'duration': None,
            'transcript': None,
            'language': None,
            'speaker_count': None,
            'pitch_volatility': None,
            'silence_ratio': None,
            'max_loudness': None,
            'analysis_successful': False,
            'error': None
        }
        
        if not audio_path or not isinstance(audio_path, str):
            result['error'] = "Invalid audio_path"
            return result
        
        if not os.path.isfile(audio_path):
            result['error'] = f"File not found: {audio_path}"
            logger.warning(f"Audio file not found: {audio_path}")
            return result
        
        try:
            if librosa is None or np is None:
                result['error'] = "Missing dependency: librosa/numpy not available"
                return result

            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            result["duration"] = round(duration, 2)
            
            # Extract bio-signals
            bio_signals = self._extract_bio_signals(y, sr)
            result.update(bio_signals)
            
            # Transcribe audio - load model lazily
            whisper_model = _load_whisper_model()
            if whisper_model:
                transcript = self._transcribe_audio(audio_path, whisper_model)
                result.update(transcript)
            
            # Perform speaker diarization - load model lazily
            if duration > 5:  # Skip very short audios
                diarization_pipeline = _load_diarization_pipeline()
                if diarization_pipeline:
                    diarization = self._diarize_speakers(audio_path, diarization_pipeline)
                    result["speaker_count"] = diarization.get("speaker_count", 0)
            
            result["analysis_successful"] = True
            
        except Exception as e:
            logger.error(f"Error analyzing audio {audio_path}: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result
    
    def _extract_bio_signals(self, y, sr: int) -> Dict:
        """Extract biological signals from audio waveform."""
        results = {
            "pitch_volatility": 0.0,
            "silence_ratio": 0.0,
            "max_loudness": 0.0
        }
        
        try:
            if librosa is None or np is None:
                return results

            # Pitch volatility (jitter)
            f0, _, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
            if not np.isnan(f0).all():
                results["pitch_volatility"] = round(float(np.nanstd(f0)), 2)
            
            # Silence ratio (pauses)
            non_silent = librosa.effects.split(y, top_db=20)
            silence_duration = (len(y) / sr) - (sum([i[1]-i[0] for i in non_silent]) / sr)
            total_duration = librosa.get_duration(y=y, sr=sr)
            results["silence_ratio"] = round(silence_duration / total_duration if total_duration > 0 else 0, 2)
            
            # Maximum loudness (RMS energy)
            rms = librosa.feature.rms(y=y)
            results["max_loudness"] = round(float(np.max(rms)), 4)
            
        except Exception as e:
            logger.warning(f"Error extracting bio-signals: {e}")
        
        return results
    
    def _transcribe_audio(self, audio_path: str, model) -> Dict:
        """Transcribe audio to text using Whisper."""
        try:
            result = model.transcribe(audio_path)
            return {
                "transcript": result["text"].strip(),
                "language": result["language"]
            }
        except Exception as e:
            logger.warning(f"Error transcribing audio: {e}")
            return {"transcript": "[Transcription failed]", "language": "unknown"}
    
    def _diarize_speakers(self, audio_path: str, pipeline) -> Dict:
        """Identify different speakers in the audio."""
        try:
            diarization = pipeline(audio_path)
            speakers = set(segment[2] for segment in diarization.itertracks(yield_label=True))
            return {"speaker_count": len(speakers)}
        except Exception as e:
            logger.warning(f"Error in speaker diarization: {e}")
            return {"speaker_count": 0}

# Global instance for easy importing
audio_analyzer = AudioAnalyzer()

def analyze_audio_file(audio_path: str) -> Dict:
    """
    Analyze an audio file for forensic purposes.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing analysis results
    """
    return audio_analyzer.analyze_audio(audio_path)
