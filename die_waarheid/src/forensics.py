"""
Audio Forensics Engine for Die Waarheid
Analyzes voice notes for bio-signal detection and stress indicators
"""

import logging
import gc
import os
import psutil
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List, Callable, Any

import numpy as np
from pathlib import Path
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

# Memory monitoring settings
MEMORY_THRESHOLD_MB = int(os.getenv("MEMORY_THRESHOLD_MB", "1000"))  # 1GB default
ENABLE_MEMORY_MONITORING = os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true"
GC_COLLECTION_INTERVAL = int(os.getenv("GC_COLLECTION_INTERVAL", "10"))  # Every 10 operations

# Global memory tracking
_memory_stats = {
    'peak_usage_mb': 0,
    'current_usage_mb': 0,
    'operations_count': 0,
    'gc_collections': 0,
    'memory_warnings': 0
}
_memory_lock = threading.Lock()

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0

def update_memory_stats():
    """Update global memory statistics"""
    global _memory_stats
    current_mb = get_memory_usage()
    
    with _memory_lock:
        _memory_stats['current_usage_mb'] = current_mb
        _memory_stats['peak_usage_mb'] = max(_memory_stats['peak_usage_mb'], current_mb)
        _memory_stats['operations_count'] += 1
        
        # Trigger garbage collection if needed
        if (_memory_stats['operations_count'] % GC_COLLECTION_INTERVAL == 0 or 
            current_mb > MEMORY_THRESHOLD_MB):
            gc.collect()
            _memory_stats['gc_collections'] += 1
            
            # Log warning if memory usage is high
            if current_mb > MEMORY_THRESHOLD_MB:
                _memory_stats['memory_warnings'] += 1
                logger.warning(f"High memory usage detected: {current_mb:.1f}MB")

def get_memory_stats() -> Dict[str, Any]:
    """Get memory usage statistics"""
    with _memory_lock:
        return {
            **_memory_stats,
            'current_usage_mb': get_memory_usage(),
            'memory_threshold_mb': MEMORY_THRESHOLD_MB,
            'monitoring_enabled': ENABLE_MEMORY_MONITORING
        }

@contextmanager
def memory_managed_operation(operation_name: str = "audio_operation"):
    """Context manager for memory-managed operations"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        if ENABLE_MEMORY_MONITORING:
            logger.debug(f"Starting {operation_name} - Memory: {start_memory:.1f}MB")
        
        yield
        
    finally:
        end_memory = get_memory_usage()
        duration = time.time() - start_time
        memory_delta = end_memory - start_memory
        
        if ENABLE_MEMORY_MONITORING:
            logger.debug(f"Completed {operation_name} - Duration: {duration:.2f}s, "
                        f"Memory: {end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
        
        update_memory_stats()
        
        # Force garbage collection if memory increased significantly
        if memory_delta > 100:  # 100MB threshold
            gc.collect()

def optimize_array_memory(arr: np.ndarray) -> np.ndarray:
    """Optimize numpy array memory usage"""
    if arr is None or arr.size == 0:
        return arr
    
    # Convert to most efficient dtype
    if arr.dtype == np.float64:
        # Check if we can safely convert to float32
        if np.allclose(arr, arr.astype(np.float32), rtol=1e-6):
            return arr.astype(np.float32)
    
    # Ensure array is contiguous for better memory access
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    
    return arr


class ForensicsEngine:
    """
    Memory-optimized audio forensics analysis engine
    Detects bio-signals: pitch volatility, silence ratio, intensity, MFCC variance,
    zero crossing rate, and spectral centroid with advanced memory management
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, use_cache: bool = True):
        self.sample_rate = sample_rate
        self.audio_data = None
        self.filename = None
        self.cache = AnalysisCache() if use_cache else None
        
        # Memory management
        self._audio_buffer_refs = weakref.WeakSet()  # Track audio buffers
        self._analysis_cache = {}  # Local analysis cache
        self._max_cache_size = 10  # Limit cache size
        
        # Performance tracking
        self._operation_count = 0
        self._last_cleanup = time.time()

    def load_audio(self, file_path: Path) -> Tuple[bool, str]:
        """
        Load audio file with memory-optimized processing and automatic cleanup

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (success: bool, message: str)
        """
        with memory_managed_operation(f"load_audio_{Path(file_path).name}"):
            try:
                file_path = Path(file_path)
                
                if not file_path.exists():
                    logger.error(f"Audio file not found: {file_path}")
                    return False, f"File not found: {file_path}"

                # Clear previous audio data to free memory
                self._cleanup_audio_data()
                
                # Load audio with memory optimization
                self.audio_data, sr = librosa.load(
                    str(file_path),
                    sr=self.sample_rate,
                    mono=True,
                    dtype=np.float32  # Use float32 instead of float64 to save memory
                )
                
                # Optimize array memory layout
                self.audio_data = optimize_array_memory(self.audio_data)
                
                # Track audio buffer for cleanup
                self._audio_buffer_refs.add(self.audio_data)
                
                self.filename = file_path.name
                duration = len(self.audio_data) / self.sample_rate
                memory_mb = self.audio_data.nbytes / 1024 / 1024

                logger.info(f"Loaded audio file: {self.filename} "
                           f"(duration: {duration:.2f}s, memory: {memory_mb:.1f}MB)")
                return True, f"Successfully loaded {self.filename}"

            except Exception as e:
                logger.error(f"Error loading audio file: {str(e)}")
                self._cleanup_audio_data()  # Cleanup on error
                return False, f"Error loading audio: {str(e)}"

    def _cleanup_audio_data(self):
        """Clean up previous audio data to free memory"""
        if self.audio_data is not None:
            # Clear reference and force garbage collection
            self.audio_data = None
            gc.collect()
            logger.debug("Cleaned up previous audio data")

    def _cleanup_cache(self):
        """Clean up analysis cache if it gets too large"""
        if len(self._analysis_cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._analysis_cache.keys())[:-self._max_cache_size//2]
            for key in keys_to_remove:
                del self._analysis_cache[key]
            gc.collect()
            logger.debug(f"Cleaned up analysis cache, removed {len(keys_to_remove)} entries")

    def _should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        self._operation_count += 1
        current_time = time.time()
        
        # Cleanup every 10 operations or every 5 minutes
        if (self._operation_count % 10 == 0 or 
            current_time - self._last_cleanup > 300):
            self._last_cleanup = current_time
            return True
        return False

    def extract_pitch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch contour using PYIN algorithm with memory optimization

        Returns:
            Tuple of (f0: pitch values, times: time frames)
        """
        if self.audio_data is None:
            logger.warning("No audio data loaded")
            return np.array([]), np.array([])

        with memory_managed_operation("extract_pitch"):
            try:
                # Check cache first
                cache_key = f"pitch_{hash(self.audio_data.tobytes())}"
                if cache_key in self._analysis_cache:
                    logger.debug("Using cached pitch extraction")
                    return self._analysis_cache[cache_key]

                f0, voiced_flag, voiced_probs = librosa.pyin(
                    self.audio_data,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate
                )

                times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sample_rate)
                
                # Optimize memory usage
                f0 = optimize_array_memory(f0)
                times = optimize_array_memory(times)
                
                # Cache result
                result = (f0, times)
                self._analysis_cache[cache_key] = result
                
                # Cleanup if needed
                if self._should_cleanup():
                    self._cleanup_cache()
                
                logger.debug(f"Extracted pitch contour with {len(f0)} frames")
                return result

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
        Calculate ratio of silence to total duration with memory optimization
        High silence ratio indicates cognitive load or hesitation

        Args:
            None (uses self.audio_data)

        Returns:
            Silence ratio (0-1)
        """
        if self.audio_data is None:
            return 0.0

with memory_managed_operation("calculate_silence_ratio"):
    try:
        # Check cache first
        cache_key = f"silence_{hash(self.audio_data.tobytes())}"
        if cache_key in self._analysis_cache:
            logger.debug("Using cached silence ratio")
            return self._analysis_cache[cache_key]

        S = librosa.feature.melspectrogram(
            y=self.audio_data,
            sr=self.sample_rate
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Optimize memory usage
        S_db = optimize_array_memory(S_db)

        threshold = np.percentile(S_db, 10)
        silent_frames = np.sum(np.mean(S_db, axis=0) < threshold)
        total_frames = S_db.shape[1]

        silence_ratio = silent_frames / total_frames if total_frames > 0 else 0.0

        # Cache result
        self._analysis_cache[cache_key] = float(silence_ratio)

        # Cleanup if needed
        if self._should_cleanup():
            self._cleanup_cache()

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

    def _extract_pitch_features(self) -> Tuple[float, float]:
        """Extract pitch features (mean and standard deviation)"""
        if self.audio_data is None:
            return 0.0, 0.0
        
        f0, voiced_flag = self.extract_pitch()
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) == 0:
            return 0.0, 0.0
        
        return float(np.mean(voiced_f0)), float(np.std(voiced_f0))

    def _extract_silence_ratio(self) -> float:
        """Extract silence ratio"""
        return self.calculate_silence_ratio()

    def _extract_intensity_features(self) -> float:
        """Extract intensity features"""
        intensity_data = self.calculate_intensity()
        return intensity_data.get('max', 0.0)

    def _extract_spectral_features(self) -> float:
        """Extract spectral features"""
        return self.calculate_spectral_centroid()


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
