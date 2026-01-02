"""
Background sound analyzer for Die Waarheid
Identifies and classifies background sounds in audio recordings.
"""

import numpy as np
import librosa
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import logging


class BackgroundSoundAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Sound classification thresholds
        self.thresholds = {
            "silence": {"energy_min": 0.0, "energy_max": 0.01},
            "quiet": {"energy_min": 0.01, "energy_max": 0.05},
            "normal": {"energy_min": 0.05, "energy_max": 0.2},
            "loud": {"energy_min": 0.2, "energy_max": 0.5},
            "very_loud": {"energy_min": 0.5, "energy_max": 1.0}
        }
        
        # Frequency bands for sound type identification
        self.freq_bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "high": (4000, 6000),
            "very_high": (6000, 20000)
        }
        
        # Sound type patterns
        self.sound_patterns = {
            "traffic": {
                "freq_profile": {"bass": True, "low_mid": True},
                "energy_range": (0.3, 0.7),
                "description": "Continuous low-frequency noise typical of traffic"
            },
            "machinery": {
                "freq_profile": {"low_mid": True, "mid": True},
                "energy_range": (0.2, 0.6),
                "description": "Mechanical sounds with mid-range dominance"
            },
            "wind": {
                "freq_profile": {"high_mid": True, "high": True},
                "energy_range": (0.1, 0.4),
                "description": "High-frequency noise characteristic of wind"
            },
            "rain": {
                "freq_profile": {"high": True, "very_high": True},
                "energy_range": (0.2, 0.5),
                "description": "Broad high-frequency noise of rainfall"
            },
            "crowd": {
                "freq_profile": {"mid": True, "high_mid": True},
                "energy_range": (0.3, 0.8),
                "description": "Multiple voice sources creating mid-range noise"
            },
            "music": {
                "freq_profile": {"bass": True, "mid": True, "high": True},
                "energy_range": (0.2, 0.9),
                "description": "Full-frequency sound with rhythmic patterns"
            }
        }
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa."""
        try:
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        except Exception as e:
            self.logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio signal."""
        return np.sqrt(np.mean(audio ** 2))
    
    def analyze_frequency_profile(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze frequency distribution across bands."""
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0])
        
        # Analyze each frequency band
        profile = {}
        for band_name, (f_min, f_max) in self.freq_bands.items():
            # Find frequency indices for this band
            band_mask = (freqs >= f_min) & (freqs <= f_max)
            band_energy = np.mean(magnitude[band_mask, :])
            profile[band_name] = band_energy
        
        # Normalize profile
        max_energy = max(profile.values())
        if max_energy > 0:
            for band in profile:
                profile[band] /= max_energy
        
        return profile
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect voice activity segments."""
        # Simple voice activity detection based on energy and frequency
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)   # 10ms hop
        
        # Compute short-time energy
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        energy = np.sum(magnitude ** 2, axis=0)
        
        # Energy threshold for voice detection
        energy_threshold = np.percentile(energy, 70)
        
        # Find voice segments
        voice_frames = energy > energy_threshold
        voice_segments = []
        
        in_voice = False
        start_frame = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_voice:
                # Voice starts
                in_voice = True
                start_frame = i
            elif not is_voice and in_voice:
                # Voice ends
                in_voice = False
                start_time = start_frame * hop_length / sr
                end_time = i * hop_length / sr
                voice_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time
                })
        
        return voice_segments
    
    def identify_sound_type(self, energy: float, freq_profile: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify likely background sound types."""
        identified = []
        
        for sound_type, pattern in self.sound_patterns.items():
            # Check energy range
            min_energy, max_energy = pattern["energy_range"]
            if not (min_energy <= energy <= max_energy):
                continue
            
            # Check frequency profile
            profile_match = True
            for band, should_be_present in pattern["freq_profile"].items():
                if should_be_present and freq_profile.get(band, 0) < 0.3:
                    profile_match = False
                    break
                if not should_be_present and freq_profile.get(band, 0) > 0.5:
                    profile_match = False
                    break
            
            if profile_match:
                confidence = min(
                    1.0 - abs(energy - np.mean([min_energy, max_energy])) / (max_energy - min_energy),
                    min(freq_profile.get(band, 0) for band in pattern["freq_profile"] if pattern["freq_profile"][band])
                )
                
                identified.append({
                    "type": sound_type,
                    "confidence": confidence,
                    "description": pattern["description"]
                })
        
        # Sort by confidence
        identified.sort(key=lambda x: x["confidence"], reverse=True)
        
        return identified
    
    def analyze_background_sounds(self, file_path: str) -> Dict[str, Any]:
        """Complete background sound analysis for an audio file."""
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            
            # Calculate overall energy
            energy = self.calculate_energy(audio)
            
            # Determine energy level category
            energy_level = "unknown"
            for level, thresholds in self.thresholds.items():
                if thresholds["energy_min"] <= energy <= thresholds["energy_max"]:
                    energy_level = level
                    break
            
            # Analyze frequency profile
            freq_profile = self.analyze_frequency_profile(audio, sr)
            
            # Detect voice activity
            voice_activity = self.detect_voice_activity(audio, sr)
            
            # Identify sound types
            sound_types = self.identify_sound_type(energy, freq_profile)
            
            # Calculate background noise characteristics
            background_segments = []
            if voice_activity:
                # Identify gaps between voice segments
                for i in range(len(voice_activity) - 1):
                    gap_start = voice_activity[i]["end"]
                    gap_end = voice_activity[i + 1]["start"]
                    if gap_end - gap_start > 0.5:  # Gap longer than 0.5 seconds
                        gap_audio = audio[int(gap_start * sr):int(gap_end * sr)]
                        gap_energy = self.calculate_energy(gap_audio)
                        background_segments.append({
                            "start": gap_start,
                            "end": gap_end,
                            "duration": gap_end - gap_start,
                            "energy": gap_energy
                        })
            
            result = {
                "file_path": file_path,
                "energy_level": energy_level,
                "energy_value": energy,
                "frequency_profile": freq_profile,
                "voice_activity": voice_activity,
                "background_segments": background_segments,
                "identified_sounds": sound_types[:3],  # Top 3 matches
                "analysis_timestamp": str(Path(file_path).stat().st_mtime)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze background sounds for {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "status": "failed"
            }
    
    def batch_analyze(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze background sounds for multiple files."""
        results = []
        
        for file_path in file_paths:
            result = self.analyze_background_sounds(file_path)
            results.append(result)
        
        return results


def main():
    """CLI interface for background sound analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze background sounds in audio files")
    parser.add_argument("--files", nargs="+", required=True, help="Audio files to analyze")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    analyzer = BackgroundSoundAnalyzer()
    results = analyzer.batch_analyze(args.files)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        for result in results:
            print(f"\nFile: {result['file_path']}")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Energy level: {result['energy_level']}")
                print(f"  Identified sounds: {[s['type'] for s in result['identified_sounds']]}")


if __name__ == "__main__":
    main()
