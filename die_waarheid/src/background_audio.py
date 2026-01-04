"""
Background Audio Analysis System
Handles background noise detection, vocal separation, and environmental audio analysis
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import signal
from sklearn.cluster import KMeans
import json

logger = logging.getLogger(__name__)

@dataclass
class BackgroundNoiseProfile:
    """Profile of background noise characteristics"""
    noise_type: str
    frequency_range: Tuple[float, float]
    amplitude_profile: List[float]
    temporal_pattern: str
    consistency_score: float
    source_distance: str  # near, medium, far
    noise_color: str  # white, pink, brown, blue
    
@dataclass
class BackgroundVocalSegment:
    """Detected background vocal segment"""
    start_time: float
    end_time: float
    confidence: float
    speaker_count: int
    gender_profile: str
    emotional_tone: str
    transcription: str
    clarity_score: float

@dataclass
class EnvironmentalContext:
    """Environmental audio context analysis"""
    location_type: str  # indoor, outdoor, vehicle, office, etc.
    room_size: str  # small, medium, large
    reverb_characteristics: Dict[str, float]
    ambient_noise_level: float
    specific_sounds: List[str]
    crowd_density: str  # empty, sparse, moderate, dense

class BackgroundAudioAnalyzer:
    """Advanced background audio analysis system"""
    
    def __init__(self):
        self.noise_profiles = self._load_noise_profiles()
        self.vocal_models = self._load_vocal_models()
        
    def _load_noise_profiles(self) -> Dict[str, Dict]:
        """Load predefined noise profiles"""
        return {
            "hvac": {"freq_range": (50, 200), "color": "brown", "pattern": "steady"},
            "traffic": {"freq_range": (100, 2000), "color": "pink", "pattern": "variable"},
            "crowd": {"freq_range": (500, 4000), "color": "pink", "pattern": "dynamic"},
            "electronics": {"freq_range": (1000, 8000), "color": "white", "pattern": "steady"},
            "nature": {"freq_range": (200, 8000), "color": "pink", "pattern": "organic"},
            "silence": {"freq_range": (20, 100), "color": "brown", "pattern": "minimal"}
        }
    
    def _load_vocal_models(self) -> Dict[str, Dict]:
        """Load background vocal detection models"""
        return {
            "male_vocal": {"freq_range": (85, 180), "formants": [730, 1090, 2440]},
            "female_vocal": {"freq_range": (165, 255), "formants": [850, 1220, 2810]},
            "child_vocal": {"freq_range": (250, 400), "formants": [900, 1300, 3000]},
            "crowd_vocal": {"freq_range": (200, 4000), "formants": "variable"}
        }
    
    def analyze_background_audio(self, audio_file: Path) -> Dict:
        """
        Comprehensive background audio analysis
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete background analysis results
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_file), sr=44100)
            
            # Analyze components
            noise_profile = self._detect_background_noise(y, sr)
            vocal_segments = self._detect_background_vocals(y, sr)
            environmental = self._analyze_environmental_context(y, sr)
            separation_quality = self._evaluate_audio_separation(y, sr)
            
            return {
                "file_path": str(audio_file),
                "analysis_timestamp": datetime.now().isoformat(),
                "duration": len(y) / sr,
                "background_noise": noise_profile,
                "background_vocals": vocal_segments,
                "environmental_context": environmental,
                "audio_separation": separation_quality,
                "overall_clarity": self._calculate_overall_clarity(noise_profile, vocal_segments, environmental)
            }
            
        except Exception as e:
            logger.error(f"Background audio analysis failed: {e}")
            return {"error": str(e)}
    
    def _detect_background_noise(self, y: np.ndarray, sr: int) -> List[BackgroundNoiseProfile]:
        """Detect and classify background noise"""
        noise_profiles = []
        
        # Perform noise detection in segments
        segment_length = sr * 10  # 10-second segments
        for i in range(0, len(y), segment_length):
            segment = y[i:i+segment_length]
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            # Frequency analysis
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # Detect dominant noise type
            detected_noise = self._classify_noise_type(freqs, magnitude, sr)
            if detected_noise:
                noise_profiles.append(detected_noise)
        
        return noise_profiles
    
    def _classify_noise_type(self, freqs: np.ndarray, magnitude: np.ndarray, sr: int) -> Optional[BackgroundNoiseProfile]:
        """Classify the type of background noise"""
        # Focus on relevant frequency range
        valid_freqs = (freqs >= 20) & (freqs <= 8000)
        freqs_filtered = freqs[valid_freqs]
        magnitude_filtered = magnitude[valid_freqs]
        
        if len(magnitude_filtered) == 0:
            return None
        
        # Calculate spectral characteristics
        spectral_centroid = np.sum(freqs_filtered * magnitude_filtered) / np.sum(magnitude_filtered)
        spectral_rolloff = self._calculate_spectral_rolloff(freqs_filtered, magnitude_filtered)
        spectral_bandwidth = self._calculate_spectral_bandwidth(freqs_filtered, magnitude_filtered)
        
        # Match against noise profiles
        best_match = None
        best_score = 0
        
        for noise_type, profile in self.noise_profiles.items():
            score = self._match_noise_profile(
                spectral_centroid, spectral_rolloff, spectral_bandwidth,
                profile, freqs_filtered, magnitude_filtered
            )
            if score > best_score and score > 0.3:  # Threshold for detection
                best_score = score
                best_match = noise_type
        
        if best_match:
            return BackgroundNoiseProfile(
                noise_type=best_match,
                frequency_range=self.noise_profiles[best_match]["freq_range"],
                amplitude_profile=magnitude_filtered.tolist()[:100],  # Sample for storage
                temporal_pattern=self.noise_profiles[best_match]["pattern"],
                consistency_score=best_score,
                source_distance=self._estimate_source_distance(magnitude_filtered),
                noise_color=self.noise_profiles[best_match]["color"]
            )
        
        return None
    
    def _detect_background_vocals(self, y: np.ndarray, sr: int) -> List[BackgroundVocalSegment]:
        """Detect background vocal segments"""
        vocal_segments = []
        
        # Use vocal detection algorithm
        # Separate harmonic/percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Detect vocal activity
        vocal_activity = self._detect_vocal_activity(y_harmonic, sr)
        
        # Group consecutive vocal frames into segments
        segments = self._group_vocal_segments(vocal_activity, sr)
        
        for segment in segments:
            start_time, end_time = segment
            segment_audio = y[int(start_time * sr):int(end_time * sr)]
            
            # Analyze vocal characteristics
            vocal_analysis = self._analyze_vocal_characteristics(segment_audio, sr)
            
            if vocal_analysis["confidence"] > 0.5:  # Threshold for detection
                vocal_segments.append(BackgroundVocalSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=vocal_analysis["confidence"],
                    speaker_count=vocal_analysis["speaker_count"],
                    gender_profile=vocal_analysis["gender"],
                    emotional_tone=vocal_analysis["emotion"],
                    transcription=vocal_analysis["transcription"],
                    clarity_score=vocal_analysis["clarity"]
                ))
        
        return vocal_segments
    
    def _detect_vocal_activity(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect vocal activity using energy and spectral features"""
        # Calculate short-time energy
        frame_length = 2048
        hop_length = 512
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        
        # Combine features for vocal detection
        vocal_activity = np.zeros_like(rms)
        
        # Vocal typically has higher energy, specific spectral range, and moderate ZCR
        energy_threshold = np.percentile(rms, 70)
        centroid_threshold = np.percentile(spectral_centroids, 40)
        centroid_max = np.percentile(spectral_centroids, 80)
        zcr_threshold = np.percentile(zcr, 60)
        
        vocal_mask = (
            (rms > energy_threshold) &
            (spectral_centroids > centroid_threshold) &
            (spectral_centroids < centroid_max) &
            (zcr < zcr_threshold)
        )
        
        vocal_activity[vocal_mask] = 1
        
        # Smooth the activity detection
        vocal_activity = self._smooth_activity_detection(vocal_activity)
        
        return vocal_activity
    
    def _analyze_environmental_context(self, y: np.ndarray, sr: int) -> EnvironmentalContext:
        """Analyze environmental context from audio"""
        # Room acoustics analysis
        reverb_analysis = self._analyze_reverberation(y, sr)
        
        # Ambient noise level
        ambient_level = self._calculate_ambient_noise_level(y)
        
        # Specific sound detection
        specific_sounds = self._detect_specific_sounds(y, sr)
        
        # Crowd density estimation
        crowd_density = self._estimate_crowd_density(y, sr)
        
        # Location type inference
        location_type = self._infer_location_type(reverb_analysis, ambient_level, specific_sounds)
        
        # Room size estimation
        room_size = self._estimate_room_size(reverb_analysis)
        
        return EnvironmentalContext(
            location_type=location_type,
            room_size=room_size,
            reverb_characteristics=reverb_analysis,
            ambient_noise_level=ambient_level,
            specific_sounds=specific_sounds,
            crowd_density=crowd_density
        )
    
    def _analyze_reverberation(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze reverberation characteristics"""
        # Calculate RT60 (reverberation time)
        rt60 = self._calculate_rt60(y, sr)
        
        # Early decay time
        edt = self._calculate_edt(y, sr)
        
        # Clarity index
        clarity = self._calculate_clarity_index(y, sr)
        
        return {
            "rt60": rt60,
            "early_decay_time": edt,
            "clarity_index": clarity,
            "reverberation_strength": self._calculate_reverb_strength(y, sr)
        }
    
    def _calculate_rt60(self, y: np.ndarray, sr: int) -> float:
        """Calculate RT60 (time for sound to decay 60dB)"""
        # Simplified RT60 calculation
        # In practice, this would use impulse response measurement
        energy = np.abs(y)**2
        
        # Find decay rate
        decay_rate = np.gradient(np.log(energy + 1e-10))
        
        # Estimate RT60 from decay rate
        avg_decay_rate = np.mean(decay_rate[decay_rate < 0])  # Only negative slopes
        
        if avg_decay_rate != 0:
            rt60 = -60 / avg_decay_rate / sr
            return max(0.1, min(rt60, 5.0))  # Clamp to reasonable range
        
        return 0.5  # Default value
    
    def _detect_specific_sounds(self, y: np.ndarray, sr: int) -> List[str]:
        """Detect specific sounds in the audio"""
        detected_sounds = []
        
        # Use spectral analysis to detect common sounds
        # This is a simplified implementation
        # In practice, you'd use pre-trained models for sound classification
        
        # Detect phone rings (high frequency bursts)
        if self._detect_phone_ring(y, sr):
            detected_sounds.append("phone_ring")
        
        # Detect door bells
        if self._detect_door_bell(y, sr):
            detected_sounds.append("door_bell")
        
        # Detect sirens
        if self._detect_siren(y, sr):
            detected_sounds.append("siren")
        
        # Detect music
        if self._detect_music(y, sr):
            detected_sounds.append("music")
        
        return detected_sounds
    
    def _evaluate_audio_separation(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Evaluate the quality of audio separation between foreground and background"""
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate separation metrics
        foreground_energy = np.sum(y_harmonic**2)
        background_energy = np.sum(y_percussive**2)
        total_energy = foreground_energy + background_energy
        
        # Signal-to-background ratio
        sbr = 10 * np.log10(foreground_energy / (background_energy + 1e-10))
        
        # Separation clarity
        separation_clarity = self._calculate_separation_clarity(y_harmonic, y_percussive)
        
        return {
            "signal_to_background_ratio": sbr,
            "separation_clarity": separation_clarity,
            "foreground_dominance": foreground_energy / total_energy,
            "background_dominance": background_energy / total_energy
        }
    
    # Helper methods (simplified implementations)
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate spectral rolloff frequency"""
        cumulative_energy = np.cumsum(magnitude)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return 0
    
    def _calculate_spectral_bandwidth(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        return bandwidth
    
    def _match_noise_profile(self, centroid: float, rolloff: float, bandwidth: float, 
                            profile: Dict, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Match audio characteristics against noise profile"""
        score = 0.0
        
        # Check frequency range match
        freq_min, freq_max = profile["freq_range"]
        if freq_min <= centroid <= freq_max:
            score += 0.4
        
        # Check spectral characteristics
        expected_rolloff = (freq_min + freq_max) * 0.85
        if abs(rolloff - expected_rolloff) < expected_rolloff * 0.3:
            score += 0.3
        
        # Check noise color characteristics
        if profile["color"] == "white" and bandwidth > 2000:
            score += 0.3
        elif profile["color"] == "pink" and 1000 < bandwidth < 3000:
            score += 0.3
        elif profile["color"] == "brown" and bandwidth < 1500:
            score += 0.3
        
        return score
    
    def _estimate_source_distance(self, magnitude: np.ndarray) -> str:
        """Estimate distance of noise source based on amplitude characteristics"""
        avg_amplitude = np.mean(magnitude)
        
        if avg_amplitude > 0.1:
            return "near"
        elif avg_amplitude > 0.05:
            return "medium"
        else:
            return "far"
    
    def _smooth_activity_detection(self, activity: np.ndarray) -> np.ndarray:
        """Smooth vocal activity detection to reduce false positives"""
        # Apply median filter to smooth activity
        window_size = 5
        smoothed = np.zeros_like(activity)
        
        for i in range(len(activity)):
            start = max(0, i - window_size // 2)
            end = min(len(activity), i + window_size // 2 + 1)
            smoothed[i] = np.median(activity[start:end])
        
        return smoothed
    
    def _group_vocal_segments(self, activity: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Group consecutive vocal frames into segments"""
        segments = []
        hop_length = 512
        frame_duration = hop_length / sr
        
        in_segment = False
        segment_start = 0
        
        for i, is_vocal in enumerate(activity):
            if is_vocal and not in_segment:
                # Start new segment
                segment_start = i * frame_duration
                in_segment = True
            elif not is_vocal and in_segment:
                # End segment
                segment_end = i * frame_duration
                if segment_end - segment_start > 0.5:  # Minimum 0.5 seconds
                    segments.append((segment_start, segment_end))
                in_segment = False
        
        # Handle case where audio ends with vocal activity
        if in_segment:
            segment_end = len(activity) * frame_duration
            if segment_end - segment_start > 0.5:
                segments.append((segment_start, segment_end))
        
        return segments
    
    def _analyze_vocal_characteristics(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze characteristics of vocal segment"""
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # Determine gender based on pitch range
        if pitch_values:
            avg_pitch = np.mean(pitch_values)
            if avg_pitch < 165:
                gender = "male"
            elif avg_pitch < 255:
                gender = "female"
            else:
                gender = "child"
        else:
            gender = "unknown"
        
        # Estimate speaker count (simplified)
        speaker_count = 1  # Default to single speaker
        
        # Emotional tone detection (simplified)
        energy = np.sum(y**2)
        if energy > np.percentile(np.abs(y)**2, 80):
            emotion = "excited"
        elif energy < np.percentile(np.abs(y)**2, 30):
            emotion = "calm"
        else:
            emotion = "neutral"
        
        return {
            "confidence": 0.7,  # Simplified confidence
            "speaker_count": speaker_count,
            "gender": gender,
            "emotion": emotion,
            "transcription": "[background speech detected]",  # Placeholder
            "clarity": 0.6  # Simplified clarity score
        }
    
    def _calculate_ambient_noise_level(self, y: np.ndarray) -> float:
        """Calculate ambient noise level in dB"""
        # Use RMS to estimate noise level
        rms = np.sqrt(np.mean(y**2))
        noise_db = 20 * np.log10(rms + 1e-10)
        return noise_db
    
    def _estimate_crowd_density(self, y: np.ndarray, sr: int) -> str:
        """Estimate crowd density from audio characteristics"""
        # Analyze spectral complexity
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # High percussive content suggests crowd
        percussive_ratio = np.sum(y_percussive**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2))
        
        if percussive_ratio > 0.7:
            return "dense"
        elif percussive_ratio > 0.4:
            return "moderate"
        elif percussive_ratio > 0.2:
            return "sparse"
        else:
            return "empty"
    
    def _infer_location_type(self, reverb: Dict, noise_level: float, sounds: List[str]) -> str:
        """Infer location type from acoustic characteristics"""
        rt60 = reverb["rt60"]
        
        if rt60 > 2.0:
            location = "large_hall"
        elif rt60 > 1.0:
            location = "medium_room"
        elif "phone_ring" in sounds or "door_bell" in sounds:
            location = "office"
        elif "siren" in sounds:
            location = "outdoor"
        elif noise_level < -40:
            location = "quiet_room"
        else:
            location = "indoor"
        
        return location
    
    def _estimate_room_size(self, reverb: Dict) -> str:
        """Estimate room size from reverberation"""
        rt60 = reverb["rt60"]
        
        if rt60 > 2.5:
            return "large"
        elif rt60 > 1.0:
            return "medium"
        else:
            return "small"
    
    def _calculate_edt(self, y: np.ndarray, sr: int) -> float:
        """Calculate early decay time"""
        # Simplified EDT calculation
        energy = np.abs(y)**2
        decay_rate = np.gradient(np.log(energy + 1e-10))
        avg_decay_rate = np.mean(decay_rate[decay_rate < 0][:10])  # First 10 frames
        
        if avg_decay_rate != 0:
            edt = -10 / avg_decay_rate / sr
            return max(0.05, min(edt, 2.0))
        
        return 0.3
    
    def _calculate_clarity_index(self, y: np.ndarray, sr: int) -> float:
        """Calculate clarity index (C80)"""
        # Simplified clarity calculation
        energy = np.abs(y)**2
        
        # Early energy (first 80ms)
        early_samples = int(0.08 * sr)
        early_energy = np.sum(energy[:early_samples])
        
        # Late energy (after 80ms)
        late_energy = np.sum(energy[early_samples:])
        
        if late_energy > 0:
            clarity = 10 * np.log10(early_energy / late_energy)
            return clarity
        
        return 0
    
    def _calculate_reverb_strength(self, y: np.ndarray, sr: int) -> float:
        """Calculate reverberation strength"""
        # Simplified reverb strength calculation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        reverb_strength = np.sum(y_percussive**2) / np.sum(y**2)
        return reverb_strength
    
    def _detect_phone_ring(self, y: np.ndarray, sr: int) -> bool:
        """Detect phone ringing sounds"""
        # Look for high frequency bursts with specific patterns
        # Simplified implementation
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        high_freq_activity = np.sum(spectral_centroids > 2000)
        return high_freq_activity > len(spectral_centroids) * 0.1
    
    def _detect_door_bell(self, y: np.ndarray, sr: int) -> bool:
        """Detect door bell sounds"""
        # Look for mid-frequency bursts
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mid_freq_activity = np.sum((spectral_centroids > 800) & (spectral_centroids < 2000))
        return mid_freq_activity > len(spectral_centroids) * 0.05
    
    def _detect_siren(self, y: np.ndarray, sr: int) -> bool:
        """Detect siren sounds"""
        # Look for frequency modulation patterns
        # Simplified implementation
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_variation = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        return pitch_variation > 200  # High variation suggests siren
    
    def _detect_music(self, y: np.ndarray, sr: int) -> bool:
        """Detect music presence"""
        # Look for harmonic structure and rhythm
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic**2) / np.sum(y**2)
        
        # Music typically has strong harmonic content
        return harmonic_ratio > 0.6
    
    def _calculate_separation_clarity(self, foreground: np.ndarray, background: np.ndarray) -> float:
        """Calculate clarity of audio separation"""
        # Calculate correlation between foreground and background
        correlation = np.corrcoef(foreground, background)[0, 1]
        
        # Lower correlation suggests better separation
        clarity = 1.0 - abs(correlation)
        return clarity
    
    def _calculate_overall_clarity(self, noise_profiles: List, vocal_segments: List, environmental: Dict) -> float:
        """Calculate overall audio clarity score"""
        clarity_score = 0.0
        
        # Noise clarity (less noise = better clarity)
        if noise_profiles:
            avg_noise_consistency = np.mean([p.consistency_score for p in noise_profiles])
            clarity_score += (1.0 - avg_noise_consistency) * 0.3
        
        # Vocal clarity
        if vocal_segments:
            avg_vocal_clarity = np.mean([v.clarity_score for v in vocal_segments])
            clarity_score += avg_vocal_clarity * 0.4
        
        # Environmental clarity
        reverb_clarity = 1.0 - (environmental.reverb_characteristics.get("rt60", 0) / 3.0)
        clarity_score += reverb_clarity * 0.3
        
        return min(1.0, clarity_score)

# Integration function for main app
def analyze_background_audio(audio_file: Path) -> Dict:
    """Main function to analyze background audio"""
    analyzer = BackgroundAudioAnalyzer()
    return analyzer.analyze_background_audio(audio_file)
