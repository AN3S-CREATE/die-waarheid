"""
Performance optimization utilities for Die Waarheid
Profiling, benchmarking, and performance monitoring
"""

import time
import functools
import logging
from typing import Callable, Any, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager for measuring execution time"""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer

        Args:
            name: Operation name for logging
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if exc_type is None:
            logger.debug(
                f"{self.name} completed",
                extra={
                    'operation': self.name,
                    'duration_seconds': self.duration,
                    'event_type': 'performance'
                }
            )
        else:
            logger.error(
                f"{self.name} failed",
                extra={
                    'operation': self.name,
                    'duration_seconds': self.duration,
                    'error': str(exc_val),
                    'event_type': 'performance_error'
                }
            )

    def get_duration(self) -> float:
        """Get duration in seconds"""
        return self.duration if self.duration else 0.0


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time

    Args:
        func: Function to measure

    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTimer(f"{func.__name__}") as timer:
            result = func(*args, **kwargs)
        return result
    return wrapper


class MemoryProfiler:
    """Memory usage profiler"""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage

        Returns:
            Dictionary with memory stats in MB
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 ** 2),
                'vms_mb': memory_info.vms / (1024 ** 2),
                'percent': process.memory_percent()
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {}

    @staticmethod
    def profile_memory(func: Callable) -> Callable:
        """
        Decorator to profile memory usage

        Args:
            func: Function to profile

        Returns:
            Wrapped function with memory profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = MemoryProfiler.get_memory_usage()
            result = func(*args, **kwargs)
            mem_after = MemoryProfiler.get_memory_usage()
            
            if mem_before and mem_after:
                delta = mem_after['rss_mb'] - mem_before['rss_mb']
                logger.debug(
                    f"{func.__name__} memory delta",
                    extra={
                        'function': func.__name__,
                        'memory_delta_mb': delta,
                        'memory_before_mb': mem_before['rss_mb'],
                        'memory_after_mb': mem_after['rss_mb'],
                        'event_type': 'memory_profile'
                    }
                )
            
            return result
        return wrapper


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: Dict[str, list] = {}
        self.start_time = datetime.now()

    def record_metric(self, metric_name: str, value: float, unit: str = ""):
        """
        Record performance metric

        Args:
            metric_name: Name of metric
            value: Metric value
            unit: Unit of measurement
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric

        Args:
            metric_name: Name of metric

        Returns:
            Dictionary with min, max, avg, count
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values),
            'total': sum(values)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {
            metric: self.get_metric_stats(metric)
            for metric in self.metrics
        }

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_time = datetime.now()


class BatchProcessor:
    """Efficient batch processing with performance optimization"""

    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize batch processor

        Args:
            batch_size: Size of each batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()

    def process_batches(
        self,
        items: list,
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> list:
        """
        Process items in batches

        Args:
            items: Items to process
            process_func: Function to process each item
            progress_callback: Optional progress callback

        Returns:
            List of processed results
        """
        results = []
        total = len(items)

        for batch_idx in range(0, total, self.batch_size):
            batch = items[batch_idx:batch_idx + self.batch_size]
            
            with PerformanceTimer(f"Batch {batch_idx // self.batch_size}") as timer:
                batch_results = [process_func(item) for item in batch]
                results.extend(batch_results)
            
            self.monitor.record_metric(
                "batch_processing_time",
                timer.duration,
                "seconds"
            )

            if progress_callback:
                progress_callback(
                    min(batch_idx + self.batch_size, total),
                    total,
                    f"Batch {batch_idx // self.batch_size}"
                )

        return results

    def get_performance_report(self) -> Dict:
        """Get performance report"""
        stats = self.monitor.get_all_stats()
        
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'metrics': stats,
            'uptime_seconds': (datetime.now() - self.monitor.start_time).total_seconds()
        }


class CacheOptimizer:
    """Cache optimization utilities"""

    @staticmethod
    def estimate_cache_size(items: int, avg_item_size_kb: float = 50) -> float:
        """
        Estimate cache size in MB

        Args:
            items: Number of items in cache
            avg_item_size_kb: Average item size in KB

        Returns:
            Estimated cache size in MB
        """
        return (items * avg_item_size_kb) / 1024

    @staticmethod
    def calculate_cache_hit_rate(hits: int, misses: int) -> float:
        """
        Calculate cache hit rate

        Args:
            hits: Number of cache hits
            misses: Number of cache misses

        Returns:
            Hit rate as percentage (0-100)
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100

    @staticmethod
    def should_evict(cache_size_mb: float, max_cache_mb: float = 500) -> bool:
        """
        Determine if cache should be evicted

        Args:
            cache_size_mb: Current cache size in MB
            max_cache_mb: Maximum allowed cache size in MB

        Returns:
            True if cache should be evicted
        """
        return cache_size_mb > max_cache_mb


class QueryOptimizer:
    """Database query optimization utilities"""

    @staticmethod
    def estimate_query_time(
        row_count: int,
        complexity: str = "simple"
    ) -> float:
        """
        Estimate query execution time

        Args:
            row_count: Number of rows to process
            complexity: Query complexity (simple, moderate, complex)

        Returns:
            Estimated time in seconds
        """
        complexity_factors = {
            'simple': 0.001,
            'moderate': 0.01,
            'complex': 0.1
        }
        
        factor = complexity_factors.get(complexity, 0.01)
        return row_count * factor / 1000

    @staticmethod
    def suggest_index(table: str, column: str, selectivity: float) -> bool:
        """
        Suggest if index should be created

        Args:
            table: Table name
            column: Column name
            selectivity: Column selectivity (0-1)

        Returns:
            True if index is recommended
        """
        return selectivity > 0.1


@contextmanager
def performance_context(operation_name: str):
    """
    Context manager for performance monitoring

    Args:
        operation_name: Name of operation
    """
    timer = PerformanceTimer(operation_name)
    timer.__enter__()
    try:
        yield timer
    finally:
        timer.__exit__(None, None, None)


if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Example: Record metrics
    for i in range(10):
        monitor.record_metric("response_time", 0.5 + i * 0.01, "seconds")
    
    # Get statistics
    stats = monitor.get_all_stats()
    print("Performance Statistics:")
    for metric, stat in stats.items():
        print(f"  {metric}: {stat}")
