"""
Audio Separation Module for Die Waarheid
Separates foreground speech from background audio
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioLayerSeparator:
    """Separates audio into foreground (speech) and background layers."""
    
    def __init__(self):
        self.sample_rate = 16000
    
    def separate_layers(self, audio_path: Path) -> Dict[str, np.ndarray]:
        """
        Separate audio into foreground and background layers.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with 'foreground' and 'background' audio arrays
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Simple separation using spectral gating
            # This is a basic implementation - in production you'd use more sophisticated methods
            S = librosa.stft(y)
            magnitude = np.abs(S)
            phase = np.angle(S)
            
            # Estimate noise floor (background)
            noise_floor = np.percentile(magnitude, 20)
            
            # Create mask for foreground (speech)
            mask = magnitude > (noise_floor * 3)
            
            # Apply mask
            foreground_S = magnitude * mask * phase
            background_S = magnitude * (~mask) * phase
            
            # Convert back to time domain
            foreground = librosa.istft(foreground_S)
            background = librosa.istft(background_S)
            
            return {
                'foreground': foreground,
                'background': background,
                'original': y
            }
            
        except Exception as e:
            logger.error(f"Error separating audio layers: {e}")
            # Return original audio as both layers if separation fails
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            return {
                'foreground': y,
                'background': np.zeros_like(y),
                'original': y
            }
    
    def get_layer_stats(self, layer: np.ndarray) -> Dict[str, float]:
        """Get statistics for an audio layer."""
        if len(layer) == 0:
            return {'rms': 0.0, 'peak': 0.0, 'duration': 0.0}
        
        rms = np.sqrt(np.mean(layer ** 2))
        peak = np.max(np.abs(layer))
        duration = len(layer) / self.sample_rate
        
        return {
            'rms': float(rms),
            'peak': float(peak),
            'duration': float(duration)
        }
