"""
Audio Forensics Engine for Die Waarheid
Analyzes voice notes for bio-signal detection and stress indicators
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import soundfile as sf

from config import (
    TARGET_SAMPLE_RATE,
    STRESS_THRESHOLD_HIGH,
    SILENCE_RATIO_THRESHOLD,
    INTENSITY_SPIKE_THRESHOLD,
    STRESS_WEIGHTS
)
from src.cache import AnalysisCache

logger = logging.getLogger(__name__)


class ForensicsEngine:
    """
    Audio forensics analysis engine
    Detects bio-signals: pitch volatility, silence ratio, intensity, MFCC variance,
    zero crossing rate, and spectral centroid
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, use_cache: bool = True):
        self.sample_rate = sample_rate
        self.audio_data = None
        self.filename = None
        self.cache = AnalysisCache() if use_cache else None

    def load_audio(self, file_path: Path) -> Tuple[bool, str]:
        """
        Load audio file with automatic format detection and resampling

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Audio file not found: {file_path}")
                return False, f"File not found: {file_path}"

            self.audio_data, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=True
            )
            self.filename = file_path.name

            logger.info(f"Loaded audio file: {self.filename} (duration: {len(self.audio_data) / self.sample_rate:.2f}s)")
            return True, f"Successfully loaded {self.filename}"

        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            return False, f"Error loading audio: {str(e)}"

    def extract_pitch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch contour using PYIN algorithm

        Returns:
            Tuple of (f0: pitch values, times: time frames)
        """
        if self.audio_data is None:
            logger.warning("No audio data loaded")
            return np.array([]), np.array([])

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                self.audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sample_rate)
            
            logger.debug(f"Extracted pitch contour with {len(f0)} frames")
            return f0, times

        except Exception as e:
            logger.error(f"Error extracting pitch: {str(e)}")
            return np.array([]), np.array([])

    def calculate_pitch_volatility(self, f0: np.ndarray) -> float:
        """
        Calculate pitch volatility (standard deviation of pitch changes)
        High volatility indicates stress or emotional arousal

        Args:
            f0: Pitch contour array

        Returns:
            Pitch volatility score (0-100)
        """
        if len(f0) < 2:
            return 0.0

        try:
            valid_f0 = f0[~np.isnan(f0)]
            
            if len(valid_f0) < 2:
                return 0.0

            pitch_changes = np.diff(valid_f0)
            volatility = np.std(pitch_changes)
            
            normalized_volatility = min(100, (volatility / 50) * 100)
            
            logger.debug(f"Pitch volatility: {normalized_volatility:.2f}")
            return float(normalized_volatility)

        except Exception as e:
            logger.error(f"Error calculating pitch volatility: {str(e)}")
            return 0.0

    def calculate_silence_ratio(self) -> float:
        """
        Calculate ratio of silence to total duration
        High silence ratio indicates cognitive load or hesitation

        Args:
            None (uses self.audio_data)

        Returns:
            Silence ratio (0-1)
        """
        if self.audio_data is None:
            return 0.0

        try:
            S = librosa.feature.melspectrogram(
                y=self.audio_data,
                sr=self.sample_rate
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            
            threshold = np.percentile(S_db, 10)
            silent_frames = np.sum(np.mean(S_db, axis=0) < threshold)
            total_frames = S_db.shape[1]
            
            silence_ratio = silent_frames / total_frames if total_frames > 0 else 0.0
            
            logger.debug(f"Silence ratio: {silence_ratio:.2f}")
            return float(silence_ratio)

        except Exception as e:
            logger.error(f"Error calculating silence ratio: {str(e)}")
            return 0.0

    def calculate_intensity(self) -> Dict[str, float]:
        """
        Calculate RMS energy intensity metrics

        Returns:
            Dictionary with intensity metrics (mean, max, std)
        """
        if self.audio_data is None:
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0}

        try:
            S = librosa.feature.melspectrogram(
                y=self.audio_data,
                sr=self.sample_rate
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            
            energy = np.mean(S_db, axis=0)
            
            metrics = {
                'mean': float(np.mean(energy)),
                'max': float(np.max(energy)),
                'std': float(np.std(energy))
            }
            
            logger.debug(f"Intensity metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating intensity: {str(e)}")
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0}

    def calculate_mfcc_variance(self) -> float:
        """
        Calculate MFCC (Mel-Frequency Cepstral Coefficient) variance
        High variance indicates emotional expression or vocal strain

        Returns:
            MFCC variance score
        """
        if self.audio_data is None:
            return 0.0

        try:
            mfcc = librosa.feature.mfcc(
                y=self.audio_data,
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            mfcc_variance = np.var(mfcc)
            
            logger.debug(f"MFCC variance: {mfcc_variance:.2f}")
            return float(mfcc_variance)

        except Exception as e:
            logger.error(f"Error calculating MFCC variance: {str(e)}")
            return 0.0

    def calculate_zero_crossing_rate(self) -> float:
        """
        Calculate zero crossing rate (ZCR)
        High ZCR indicates fricative sounds and emotional intensity

        Returns:
            Mean zero crossing rate
        """
        if self.audio_data is None:
            return 0.0

        try:
            zcr = librosa.feature.zero_crossing_rate(self.audio_data)[0]
            mean_zcr = np.mean(zcr)
            
            logger.debug(f"Zero crossing rate: {mean_zcr:.4f}")
            return float(mean_zcr)

        except Exception as e:
            logger.error(f"Error calculating zero crossing rate: {str(e)}")
            return 0.0

    def calculate_spectral_centroid(self) -> float:
        """
        Calculate spectral centroid
        Indicates brightness of sound (higher = brighter/more tense)

        Returns:
            Mean spectral centroid (Hz)
        """
        if self.audio_data is None:
            return 0.0

        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=self.audio_data,
                sr=self.sample_rate
            )[0]
            
            mean_centroid = np.mean(spectral_centroids)
            
            logger.debug(f"Spectral centroid: {mean_centroid:.2f} Hz")
            return float(mean_centroid)

        except Exception as e:
            logger.error(f"Error calculating spectral centroid: {str(e)}")
            return 0.0

    def analyze(self, file_path: Path) -> Dict:
        """
        Complete forensic analysis of audio file with caching support

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with all forensic metrics
        """
        file_path = Path(file_path)
        
        if self.cache:
            cached_result = self.cache.get(file_path)
            if cached_result:
                logger.info(f"Using cached result for {file_path.name}")
                return cached_result
        
        success, message = self.load_audio(file_path)
        
        if not success:
            logger.error(f"Cannot analyze: {message}")
            return {
                'success': False,
                'message': message,
                'filename': str(file_path)
            }

        try:
            # Validate audio data before analysis
            if self.audio_data is None or len(self.audio_data) == 0:
                logger.error("Empty or invalid audio data")
                return {
                    'success': False,
                    'message': "Empty or invalid audio data",
                    'filename': str(file_path)
                }
            
            f0, times = self.extract_pitch()
            
            pitch_volatility = self.calculate_pitch_volatility(f0)
            silence_ratio = self.calculate_silence_ratio()
            intensity = self.calculate_intensity()
            mfcc_variance = self.calculate_mfcc_variance()
            zcr = self.calculate_zero_crossing_rate()
            spectral_centroid = self.calculate_spectral_centroid()
            
            duration = len(self.audio_data) / self.sample_rate
            
            stress_level = self._calculate_stress_level(
                pitch_volatility,
                silence_ratio,
                intensity['max'],
                mfcc_variance
            )
            
            result = {
                'success': True,
                'filename': self.filename,
                'duration': duration,
                'pitch_volatility': pitch_volatility,
                'silence_ratio': silence_ratio,
                'intensity': intensity,
                'mfcc_variance': mfcc_variance,
                'zero_crossing_rate': zcr,
                'spectral_centroid': spectral_centroid,
                'stress_level': stress_level,
                'stress_threshold_exceeded': stress_level > STRESS_THRESHOLD_HIGH,
                'high_cognitive_load': silence_ratio > SILENCE_RATIO_THRESHOLD
            }
            
            if self.cache:
                self.cache.set(file_path, result)
            
            logger.info(f"Analysis complete for {self.filename}: stress_level={stress_level:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return {
                'success': False,
                'message': f"Analysis error: {str(e)}",
                'filename': self.filename
            }

    def _calculate_stress_level(
        self,
        pitch_volatility: float,
        silence_ratio: float,
        intensity_max: float,
        mfcc_variance: float
    ) -> float:
        """
        Calculate composite stress level from multiple bio-signals using configurable weights

        Args:
            pitch_volatility: Pitch volatility score (0-100)
            silence_ratio: Silence ratio (0-1)
            intensity_max: Maximum intensity (typically -80 to 0 dB)
            mfcc_variance: MFCC variance

        Returns:
            Composite stress level (0-100)
        """
        try:
            normalized_intensity = max(0, min(100, (intensity_max + 80) / 80 * 100))
            normalized_mfcc = min(100, mfcc_variance)
            
            pitch_component = pitch_volatility * STRESS_WEIGHTS['pitch']
            silence_component = (silence_ratio * 100) * STRESS_WEIGHTS['silence']
            intensity_component = normalized_intensity * STRESS_WEIGHTS['intensity']
            mfcc_component = normalized_mfcc * STRESS_WEIGHTS['mfcc']
            
            stress_level = pitch_component + silence_component + intensity_component + mfcc_component
            
            logger.debug(f"Stress components - pitch: {pitch_component:.2f}, silence: {silence_component:.2f}, intensity: {intensity_component:.2f}, mfcc: {mfcc_component:.2f}")
            
            return min(100, max(0, stress_level))

        except Exception as e:
            logger.error(f"Error calculating stress level: {str(e)}")
            return 0.0

    def calculate_stress_level(self, pitch_volatility: float, silence_ratio: float, 
                              intensity_max: float, mfcc_variance: float, 
                              zero_crossing_rate: float = 0.0) -> float:
        """
        Public wrapper for stress level calculation
        
        Args:
            pitch_volatility: Pitch volatility value
            silence_ratio: Silence ratio value
            intensity_max: Maximum intensity value
            mfcc_variance: MFCC variance value
            zero_crossing_rate: Zero crossing rate (optional, for future use)
            
        Returns:
            Stress level (0-100)
        """
        # Note: zero_crossing_rate is accepted for API compatibility but not currently used
        return self._calculate_stress_level(pitch_volatility, silence_ratio, intensity_max, mfcc_variance)

    def batch_analyze(self, file_paths: List[Path], progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[Dict]:
        """
        Analyze multiple audio files sequentially

        Args:
            file_paths: List of paths to audio files
            progress_callback: Optional callback for progress updates (current, total, filename)

        Returns:
            List of analysis results
        """
        results = []
        total = len(file_paths)
        
        for idx, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(idx + 1, total, file_path.name)
            logger.info(f"Analyzing file {idx + 1}/{total}: {file_path.name}")
            result = self.analyze(file_path)
            results.append(result)
        
        logger.info(f"Batch analysis complete: {len(results)} files processed")
        return results

    def batch_analyze_parallel(self, file_paths: List[Path], max_workers: int = 4, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[Dict]:
        """
        Analyze multiple audio files in parallel

        Args:
            file_paths: List of paths to audio files
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback for progress updates

        Returns:
            List of analysis results
        """
        results = []
        total = len(file_paths)
        completed = [0]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze, fp): fp for fp in file_paths}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed[0] += 1
                    
                    if progress_callback:
                        file_path = futures[future]
                        progress_callback(completed[0], total, file_path.name)
                    
                    logger.info(f"Completed {completed[0]}/{total}: {futures[future].name}")
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    results.append({
                        'success': False,
                        'message': f'Analysis error: {str(e)}',
                        'filename': file_path.name
                    })
        
        logger.info(f"Parallel batch analysis complete: {len(results)} files processed")
        return results


if __name__ == "__main__":
    engine = ForensicsEngine()
    
    test_file = Path("data/audio/test_audio.wav")
    if test_file.exists():
        result = engine.analyze(test_file)
        print("Analysis Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"Test file not found: {test_file}")
