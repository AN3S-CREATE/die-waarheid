"""
Health Check and Monitoring for Die Waarheid
System health status, performance metrics, and diagnostics
"""

import logging
import psutil
import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from config import (
    APP_NAME,
    APP_VERSION,
    AUDIO_DIR,
    TEXT_DIR,
    TEMP_DIR,
    REPORTS_DIR,
    CREDENTIALS_DIR,
    GEMINI_API_KEY
)

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    System health checker for Die Waarheid
    Monitors system resources, dependencies, and configuration
    """

    def __init__(self):
        """Initialize health checker"""
        self.last_check = None
        self.check_results = {}

    def check_system_resources(self) -> Dict:
        """
        Check system resource availability

        Returns:
            Dictionary with resource status
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = shutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'cpu_status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical',
                'memory_available_mb': memory.available / (1024 ** 2),
                'memory_percent': memory.percent,
                'memory_status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 95 else 'critical',
                'disk_available_gb': disk.free / (1024 ** 3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_status': 'healthy' if (disk.free / disk.total) > 0.1 else 'warning' if (disk.free / disk.total) > 0.05 else 'critical'
            }

        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return {'error': str(e), 'status': 'error'}

    def check_directories(self) -> Dict:
        """
        Check required directories

        Returns:
            Dictionary with directory status
        """
        directories = {
            'audio': AUDIO_DIR,
            'text': TEXT_DIR,
            'temp': TEMP_DIR,
            'reports': REPORTS_DIR,
            'credentials': CREDENTIALS_DIR
        }

        results = {}
        for name, path in directories.items():
            try:
                path = Path(path)
                exists = path.exists()
                writable = exists and path.stat().st_mode & 0o200
                results[name] = {
                    'path': str(path),
                    'exists': exists,
                    'writable': writable,
                    'status': 'healthy' if exists and writable else 'warning'
                }
            except Exception as e:
                results[name] = {
                    'path': str(path),
                    'error': str(e),
                    'status': 'error'
                }

        return results

    def check_dependencies(self) -> Dict:
        """
        Check required dependencies

        Returns:
            Dictionary with dependency status
        """
        dependencies = {
            'librosa': 'librosa',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'google-generativeai': 'google.generativeai',
            'streamlit': 'streamlit',
            'plotly': 'plotly',
            'sqlalchemy': 'sqlalchemy',
            'pydantic': 'pydantic'
        }

        results = {}
        for package_name, import_name in dependencies.items():
            try:
                __import__(import_name)
                results[package_name] = {
                    'installed': True,
                    'status': 'healthy'
                }
            except ImportError:
                results[package_name] = {
                    'installed': False,
                    'status': 'warning'
                }

        return results

    def check_configuration(self) -> Dict:
        """
        Check configuration status

        Returns:
            Dictionary with configuration status
        """
        results = {
            'app_name': APP_NAME,
            'app_version': APP_VERSION,
            'gemini_configured': bool(GEMINI_API_KEY),
            'credentials_available': Path(CREDENTIALS_DIR).exists()
        }

        if not GEMINI_API_KEY:
            results['gemini_status'] = 'warning'
        else:
            results['gemini_status'] = 'healthy'

        return results

    def check_database(self) -> Dict:
        """
        Check database connectivity

        Returns:
            Dictionary with database status
        """
        try:
            from database import DatabaseManager

            db = DatabaseManager()
            if db.engine:
                db.close()
                return {
                    'connected': True,
                    'status': 'healthy'
                }
            else:
                return {
                    'connected': False,
                    'status': 'error'
                }

        except Exception as e:
            logger.error(f"Error checking database: {str(e)}")
            return {
                'connected': False,
                'error': str(e),
                'status': 'error'
            }

    def get_full_health_status(self) -> Dict:
        """
        Get complete system health status

        Returns:
            Dictionary with full health status
        """
        try:
            self.last_check = datetime.now()

            resources = self.check_system_resources()
            directories = self.check_directories()
            dependencies = self.check_dependencies()
            configuration = self.check_configuration()
            database = self.check_database()

            overall_status = self._determine_overall_status(
                resources, directories, dependencies, configuration, database
            )

            self.check_results = {
                'timestamp': self.last_check.isoformat(),
                'overall_status': overall_status,
                'system_resources': resources,
                'directories': directories,
                'dependencies': dependencies,
                'configuration': configuration,
                'database': database
            }

            return self.check_results

        except Exception as e:
            logger.error(f"Error getting full health status: {str(e)}")
            return {
                'error': str(e),
                'overall_status': 'error'
            }

    def _determine_overall_status(self, *checks) -> str:
        """
        Determine overall system status

        Args:
            *checks: Various check result dictionaries

        Returns:
            Overall status string
        """
        statuses = []

        for check in checks:
            if isinstance(check, dict):
                if 'status' in check:
                    statuses.append(check['status'])
                elif 'overall_status' in check:
                    statuses.append(check['overall_status'])
                else:
                    for value in check.values():
                        if isinstance(value, dict) and 'status' in value:
                            statuses.append(value['status'])

        if 'critical' in statuses:
            return 'critical'
        elif 'error' in statuses:
            return 'error'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'

    def get_status_summary(self) -> Dict:
        """
        Get brief status summary

        Returns:
            Dictionary with status summary
        """
        full_status = self.get_full_health_status()

        return {
            'timestamp': full_status.get('timestamp'),
            'overall_status': full_status.get('overall_status'),
            'cpu_percent': full_status.get('system_resources', {}).get('cpu_percent'),
            'memory_percent': full_status.get('system_resources', {}).get('memory_percent'),
            'disk_available_gb': full_status.get('system_resources', {}).get('disk_available_gb'),
            'gemini_configured': full_status.get('configuration', {}).get('gemini_configured'),
            'database_connected': full_status.get('database', {}).get('connected')
        }

    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics

        Returns:
            Dictionary with performance metrics
        """
        try:
            process = psutil.Process()

            return {
                'cpu_percent': process.cpu_percent(interval=0.1),
                'memory_mb': process.memory_info().rss / (1024 ** 2),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'error': str(e)}

    def get_diagnostics(self) -> Dict:
        """
        Get complete diagnostics information

        Returns:
            Dictionary with diagnostics
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': self.get_full_health_status(),
            'performance_metrics': self.get_performance_metrics(),
            'system_info': {
                'platform': __import__('platform').platform(),
                'python_version': __import__('sys').version,
                'processor': __import__('platform').processor()
            }
        }


class HealthEndpoint:
    """
    Health check endpoint for monitoring
    Can be integrated into Streamlit or REST API
    """

    def __init__(self):
        """Initialize health endpoint"""
        self.checker = HealthChecker()

    def get_health(self) -> Dict:
        """Get health status"""
        return self.checker.get_status_summary()

    def get_detailed_health(self) -> Dict:
        """Get detailed health status"""
        return self.checker.get_full_health_status()

    def get_diagnostics(self) -> Dict:
        """Get diagnostics"""
        return self.checker.get_diagnostics()

    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        status = self.checker.get_status_summary()
        return status.get('overall_status') in ['healthy', 'warning']


if __name__ == "__main__":
    checker = HealthChecker()
    status = checker.get_status_summary()
    print("Health Status:")
    print(f"Overall: {status.get('overall_status')}")
    print(f"CPU: {status.get('cpu_percent'):.1f}%")
    print(f"Memory: {status.get('memory_percent'):.1f}%")
    print(f"Disk Available: {status.get('disk_available_gb'):.2f} GB")
