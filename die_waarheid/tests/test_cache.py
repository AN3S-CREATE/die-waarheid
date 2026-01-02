"""
Unit tests for caching layer
Tests cache operations, hit/miss, and persistence
"""

import pytest
import tempfile
from pathlib import Path

from src.cache import AnalysisCache


class TestAnalysisCache:
    """Test AnalysisCache functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache instance with temp directory"""
        return AnalysisCache(cache_dir=temp_cache_dir)

    @pytest.fixture
    def temp_audio_file(self, temp_cache_dir):
        """Create temporary audio file"""
        audio_file = temp_cache_dir / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        return audio_file

    def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.cache is not None
        assert cache.cache_dir.exists()

    def test_get_file_hash(self, cache, temp_audio_file):
        """Test file hash generation"""
        hash1 = cache.get_file_hash(temp_audio_file)
        assert hash1 is not None
        assert len(hash1) == 32  # MD5 hash length

    def test_file_hash_consistency(self, cache, temp_audio_file):
        """Test that same file produces same hash"""
        hash1 = cache.get_file_hash(temp_audio_file)
        hash2 = cache.get_file_hash(temp_audio_file)
        assert hash1 == hash2

    def test_different_files_different_hashes(self, cache, temp_cache_dir):
        """Test that different files produce different hashes"""
        file1 = temp_cache_dir / "file1.wav"
        file2 = temp_cache_dir / "file2.wav"
        file1.write_bytes(b"data1")
        file2.write_bytes(b"data2")

        hash1 = cache.get_file_hash(file1)
        hash2 = cache.get_file_hash(file2)
        assert hash1 != hash2

    def test_cache_set_and_get(self, cache, temp_audio_file):
        """Test storing and retrieving from cache"""
        result = {
            'success': True,
            'filename': 'test.wav',
            'stress_level': 45.5
        }

        success = cache.set(temp_audio_file, result)
        assert success is True

        cached_result = cache.get(temp_audio_file)
        assert cached_result is not None
        assert cached_result['stress_level'] == 45.5

    def test_cache_miss(self, cache, temp_cache_dir):
        """Test cache miss for non-existent file"""
        non_existent = temp_cache_dir / "non_existent.wav"
        result = cache.get(non_existent)
        assert result is None

    def test_cache_clear(self, cache, temp_audio_file):
        """Test clearing cache"""
        result = {'success': True, 'stress_level': 45.5}
        cache.set(temp_audio_file, result)

        cached = cache.get(temp_audio_file)
        assert cached is not None

        cache.clear()
        cached = cache.get(temp_audio_file)
        assert cached is None

    def test_cache_with_complex_data(self, cache, temp_audio_file):
        """Test caching complex nested data"""
        result = {
            'success': True,
            'filename': 'test.wav',
            'duration': 10.5,
            'metrics': {
                'pitch_volatility': 45.2,
                'silence_ratio': 0.3,
                'intensity': {
                    'mean': -20.0,
                    'max': -5.0,
                    'std': 8.5
                }
            },
            'flags': ['high_stress', 'cognitive_load']
        }

        cache.set(temp_audio_file, result)
        cached = cache.get(temp_audio_file)

        assert cached['metrics']['pitch_volatility'] == 45.2
        assert cached['metrics']['intensity']['mean'] == -20.0
        assert 'high_stress' in cached['flags']

    def test_invalid_file_path(self, cache):
        """Test handling of invalid file path"""
        invalid_path = Path("/invalid/path/file.wav")
        hash_result = cache.get_file_hash(invalid_path)
        assert hash_result == ""

    def test_cache_persistence(self, temp_cache_dir, temp_audio_file):
        """Test cache persistence across instances"""
        cache1 = AnalysisCache(cache_dir=temp_cache_dir)
        result = {'success': True, 'stress_level': 55.0}
        cache1.set(temp_audio_file, result)
        cache1.close()

        cache2 = AnalysisCache(cache_dir=temp_cache_dir)
        cached = cache2.get(temp_audio_file)
        assert cached is not None
        assert cached['stress_level'] == 55.0
        cache2.close()

    def test_cache_close(self, cache):
        """Test cache closing"""
        cache.close()
        assert cache.cache is None or not hasattr(cache.cache, 'close')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
