"""
Unit tests for health check and monitoring
Tests system health checks, resource monitoring, and diagnostics
"""

import pytest
from src.health import HealthChecker, HealthEndpoint


class TestHealthChecker:
    """Test HealthChecker functionality"""

    @pytest.fixture
    def checker(self):
        """Create health checker instance"""
        return HealthChecker()

    def test_checker_initialization(self, checker):
        """Test health checker initialization"""
        assert checker is not None
        assert checker.last_check is None

    def test_check_system_resources(self, checker):
        """Test system resource checking"""
        resources = checker.check_system_resources()

        assert 'cpu_percent' in resources
        assert 'memory_percent' in resources
        assert 'disk_available_gb' in resources
        assert 'cpu_status' in resources
        assert 'memory_status' in resources
        assert 'disk_status' in resources

    def test_cpu_percent_valid_range(self, checker):
        """Test that CPU percent is in valid range"""
        resources = checker.check_system_resources()
        assert 0 <= resources['cpu_percent'] <= 100

    def test_memory_percent_valid_range(self, checker):
        """Test that memory percent is in valid range"""
        resources = checker.check_system_resources()
        assert 0 <= resources['memory_percent'] <= 100

    def test_disk_available_positive(self, checker):
        """Test that disk available is positive"""
        resources = checker.check_system_resources()
        assert resources['disk_available_gb'] >= 0

    def test_resource_status_values(self, checker):
        """Test that status values are valid"""
        resources = checker.check_system_resources()
        valid_statuses = ['healthy', 'warning', 'critical', 'error']

        assert resources['cpu_status'] in valid_statuses
        assert resources['memory_status'] in valid_statuses
        assert resources['disk_status'] in valid_statuses

    def test_check_directories(self, checker):
        """Test directory checking"""
        directories = checker.check_directories()

        assert 'audio' in directories
        assert 'text' in directories
        assert 'temp' in directories
        assert 'reports' in directories
        assert 'credentials' in directories

    def test_directory_status_fields(self, checker):
        """Test directory status fields"""
        directories = checker.check_directories()

        for dir_name, dir_info in directories.items():
            assert 'path' in dir_info
            assert 'status' in dir_info

    def test_check_dependencies(self, checker):
        """Test dependency checking"""
        dependencies = checker.check_dependencies()

        assert 'numpy' in dependencies
        assert 'pandas' in dependencies
        assert 'streamlit' in dependencies
        assert 'sqlalchemy' in dependencies
        assert 'pydantic' in dependencies

    def test_dependency_status_fields(self, checker):
        """Test dependency status fields"""
        dependencies = checker.check_dependencies()

        for package, info in dependencies.items():
            assert 'installed' in info
            assert 'status' in info
            assert isinstance(info['installed'], bool)

    def test_check_configuration(self, checker):
        """Test configuration checking"""
        config = checker.check_configuration()

        assert 'app_name' in config
        assert 'app_version' in config
        assert 'gemini_configured' in config
        assert 'credentials_available' in config

    def test_get_full_health_status(self, checker):
        """Test getting full health status"""
        status = checker.get_full_health_status()

        assert 'timestamp' in status
        assert 'overall_status' in status
        assert 'system_resources' in status
        assert 'directories' in status
        assert 'dependencies' in status
        assert 'configuration' in status

    def test_overall_status_values(self, checker):
        """Test that overall status is valid"""
        status = checker.get_full_health_status()
        valid_statuses = ['healthy', 'warning', 'critical', 'error']
        assert status['overall_status'] in valid_statuses

    def test_get_status_summary(self, checker):
        """Test getting status summary"""
        summary = checker.get_status_summary()

        assert 'timestamp' in summary
        assert 'overall_status' in summary
        assert 'cpu_percent' in summary
        assert 'memory_percent' in summary
        assert 'disk_available_gb' in summary
        assert 'gemini_configured' in summary

    def test_get_performance_metrics(self, checker):
        """Test getting performance metrics"""
        metrics = checker.get_performance_metrics()

        assert 'cpu_percent' in metrics
        assert 'memory_mb' in metrics
        assert 'num_threads' in metrics
        assert 'open_files' in metrics

    def test_performance_metrics_valid_values(self, checker):
        """Test that performance metrics have valid values"""
        metrics = checker.get_performance_metrics()

        assert metrics['cpu_percent'] >= 0
        assert metrics['memory_mb'] >= 0
        assert metrics['num_threads'] > 0
        assert metrics['open_files'] >= 0

    def test_get_diagnostics(self, checker):
        """Test getting complete diagnostics"""
        diagnostics = checker.get_diagnostics()

        assert 'timestamp' in diagnostics
        assert 'health_status' in diagnostics
        assert 'performance_metrics' in diagnostics
        assert 'system_info' in diagnostics

    def test_system_info_fields(self, checker):
        """Test system info fields"""
        diagnostics = checker.get_diagnostics()
        system_info = diagnostics['system_info']

        assert 'platform' in system_info
        assert 'python_version' in system_info
        assert 'processor' in system_info


class TestHealthEndpoint:
    """Test HealthEndpoint functionality"""

    @pytest.fixture
    def endpoint(self):
        """Create health endpoint instance"""
        return HealthEndpoint()

    def test_endpoint_initialization(self, endpoint):
        """Test health endpoint initialization"""
        assert endpoint is not None
        assert endpoint.checker is not None

    def test_get_health(self, endpoint):
        """Test getting health status"""
        health = endpoint.get_health()

        assert 'overall_status' in health
        assert 'cpu_percent' in health
        assert 'memory_percent' in health

    def test_get_detailed_health(self, endpoint):
        """Test getting detailed health status"""
        health = endpoint.get_detailed_health()

        assert 'overall_status' in health
        assert 'system_resources' in health
        assert 'directories' in health
        assert 'dependencies' in health

    def test_get_diagnostics(self, endpoint):
        """Test getting diagnostics"""
        diagnostics = endpoint.get_diagnostics()

        assert 'timestamp' in diagnostics
        assert 'health_status' in diagnostics
        assert 'performance_metrics' in diagnostics
        assert 'system_info' in diagnostics

    def test_is_healthy_true(self, endpoint):
        """Test is_healthy when system is healthy"""
        is_healthy = endpoint.is_healthy()
        assert isinstance(is_healthy, bool)

    def test_health_status_consistency(self, endpoint):
        """Test that health status is consistent"""
        health1 = endpoint.get_health()
        health2 = endpoint.get_health()

        assert health1['overall_status'] == health2['overall_status']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
