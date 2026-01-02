"""
Caching layer for Die Waarheid
Provides persistent caching for expensive operations
"""

import logging
import hashlib
import shelve
from pathlib import Path
from typing import Dict, Optional, Any

from config import TEMP_DIR

logger = logging.getLogger(__name__)


class AnalysisCache:
    """
    Persistent cache for forensic analysis results
    Uses shelve for simple key-value storage
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache

        Args:
            cache_dir: Directory for cache storage (uses TEMP_DIR if None)
        """
        if cache_dir is None:
            cache_dir = TEMP_DIR
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = str(self.cache_dir / "analysis_cache")
        
        try:
            self.cache = shelve.open(self.cache_path)
            logger.info(f"Initialized analysis cache at {self.cache_path}")
        except Exception as e:
            logger.error(f"Error initializing cache: {str(e)}")
            self.cache = None

    def get_file_hash(self, file_path: Path) -> str:
        """
        Generate hash for cache key from file content

        Args:
            file_path: Path to file

        Returns:
            MD5 hash of file content
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            return ""

    def get(self, file_path: Path) -> Optional[Dict]:
        """
        Get cached analysis result

        Args:
            file_path: Path to audio file

        Returns:
            Cached result or None if not found
        """
        if self.cache is None:
            return None

        try:
            cache_key = self.get_file_hash(file_path)
            if not cache_key:
                return None

            if cache_key in self.cache:
                logger.debug(f"Cache hit for {file_path.name}")
                return self.cache[cache_key]

            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(self, file_path: Path, result: Dict) -> bool:
        """
        Store analysis result in cache

        Args:
            file_path: Path to audio file
            result: Analysis result to cache

        Returns:
            True if successful
        """
        if self.cache is None:
            return False

        try:
            cache_key = self.get_file_hash(file_path)
            if not cache_key:
                return False

            self.cache[cache_key] = result
            self.cache.sync()
            logger.debug(f"Cached result for {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear all cached results

        Returns:
            True if successful
        """
        if self.cache is None:
            return False

        try:
            self.cache.clear()
            self.cache.sync()
            logger.info("Cache cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def close(self):
        """Close cache"""
        if self.cache is not None:
            try:
                self.cache.close()
                logger.info("Cache closed")
            except Exception as e:
                logger.error(f"Error closing cache: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    cache = AnalysisCache()
    print("Cache initialized successfully")
