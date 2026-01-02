"""
Error recovery and resilience utilities for Die Waarheid
Circuit breakers, fallbacks, and graceful degradation
"""

import logging
import time
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by stopping requests to failing services
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or None if circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN")
                return None

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            logger.error(f"Circuit breaker {self.name} caught exception: {str(e)}")
            raise

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED state")

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class Fallback:
    """Fallback strategy for handling failures"""

    def __init__(self, name: str):
        """
        Initialize fallback

        Args:
            name: Fallback name
        """
        self.name = name
        self.fallback_func = None
        self.fallback_value = None

    def set_function(self, func: Callable) -> 'Fallback':
        """
        Set fallback function

        Args:
            func: Function to call on failure

        Returns:
            Self for chaining
        """
        self.fallback_func = func
        return self

    def set_value(self, value: Any) -> 'Fallback':
        """
        Set fallback value

        Args:
            value: Value to return on failure

        Returns:
            Self for chaining
        """
        self.fallback_value = value
        return self

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute fallback

        Args:
            *args: Arguments for fallback function
            **kwargs: Keyword arguments for fallback function

        Returns:
            Fallback result
        """
        if self.fallback_func:
            return self.fallback_func(*args, **kwargs)
        return self.fallback_value


class RetryStrategy:
    """Configurable retry strategy"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry strategy

        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry strategy

        Args:
            func: Function to execute
            *args: Function arguments
            on_retry: Optional callback on retry
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        attempt = 0
        last_exception = None

        while attempt < self.max_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                last_exception = e

                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed, retrying in {delay:.2f}s: {str(e)}"
                    )

                    if on_retry:
                        on_retry(attempt, delay, e)

                    time.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt"""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay


class BulkheadPattern:
    """
    Bulkhead pattern for resource isolation
    Limits concurrent operations to prevent resource exhaustion
    """

    def __init__(self, name: str, max_concurrent: int = 10):
        """
        Initialize bulkhead

        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent operations
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self.queue: List[Callable] = []

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with bulkhead protection

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if self.current_count >= self.max_concurrent:
            logger.warning(
                f"Bulkhead {self.name} at capacity ({self.current_count}/{self.max_concurrent})"
            )
            raise RuntimeError(f"Bulkhead {self.name} at capacity")

        self.current_count += 1
        try:
            return func(*args, **kwargs)
        finally:
            self.current_count -= 1

    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status"""
        return {
            'name': self.name,
            'current': self.current_count,
            'max': self.max_concurrent,
            'available': self.max_concurrent - self.current_count
        }


class TimeoutHandler:
    """Handle operation timeouts"""

    @staticmethod
    def execute_with_timeout(
        func: Callable,
        timeout_seconds: float,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with timeout

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If operation exceeds timeout
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation exceeded {timeout_seconds}s timeout")

        # Set signal handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds) + 1)

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            logger.error(f"Operation timed out after {timeout_seconds}s")
            raise


class ResilientExecutor:
    """
    Combines multiple resilience patterns
    Provides comprehensive error handling and recovery
    """

    def __init__(self, name: str):
        """
        Initialize resilient executor

        Args:
            name: Executor name
        """
        self.name = name
        self.circuit_breaker = CircuitBreaker(name)
        self.retry_strategy = RetryStrategy()
        self.bulkhead = BulkheadPattern(name)
        self.fallback = Fallback(name)

    def execute(
        self,
        func: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        use_bulkhead: bool = True,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with all resilience patterns

        Args:
            func: Function to execute
            *args: Function arguments
            use_circuit_breaker: Enable circuit breaker
            use_retry: Enable retry logic
            use_bulkhead: Enable bulkhead
            timeout_seconds: Optional timeout
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback value
        """
        try:
            if use_bulkhead:
                def bulkhead_wrapper():
                    return self.bulkhead.execute(func, *args, **kwargs)
                wrapped_func = bulkhead_wrapper
            else:
                wrapped_func = lambda: func(*args, **kwargs)

            if use_retry:
                def retry_wrapper():
                    return self.retry_strategy.execute(wrapped_func)
                wrapped_func = retry_wrapper

            if use_circuit_breaker:
                result = self.circuit_breaker.call(wrapped_func)
                if result is None:
                    return self.fallback.execute(*args, **kwargs)
                return result
            else:
                return wrapped_func()

        except Exception as e:
            logger.error(f"Resilient execution failed: {str(e)}")
            return self.fallback.execute(*args, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Get executor status"""
        return {
            'name': self.name,
            'circuit_breaker': self.circuit_breaker.get_state(),
            'bulkhead': self.bulkhead.get_status()
        }


if __name__ == "__main__":
    executor = ResilientExecutor("test_executor")
    
    def test_func():
        return "Success"
    
    result = executor.execute(test_func)
    print(f"Result: {result}")
    print(f"Status: {executor.get_status()}")
