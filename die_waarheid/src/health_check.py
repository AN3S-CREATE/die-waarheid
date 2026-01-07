"""
Health check and monitoring endpoints for Die Waarheid
"""

import logging
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

from config import (
    APP_NAME,
    APP_VERSION,
    AUDIO_DIR,
    TEXT_DIR,
    TEMP_DIR,
    REPORTS_DIR
)

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check and monitoring system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_check = None
        self.health_status = "healthy"
        self.incidents = []
        
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'uptime_seconds': int(time.time() - self.start_time),
            'checks': {}
        }
        
        # Check disk space
        disk_check = self._check_disk_space()
        health_report['checks']['disk_space'] = disk_check
        
        # Check memory usage
        memory_check = self._check_memory_usage()
        health_report['checks']['memory'] = memory_check
        
        # Check directories
        dir_check = self._check_directories()
        health_report['checks']['directories'] = dir_check
        
        # Check API connectivity
        api_check = self._check_api_connectivity()
        health_report['checks']['api'] = api_check
        
        # Determine overall status
        if any(check['status'] != 'healthy' for check in health_report['checks'].values()):
            health_report['status'] = 'unhealthy'
            self.health_status = 'unhealthy'
        else:
            self.health_status = 'healthy'
        
        self.last_check = datetime.now()
        return health_report
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            status = 'healthy'
            if free_gb < 1:  # Less than 1GB free
                status = 'critical'
            elif free_gb < 5:  # Less than 5GB free
                status = 'warning'
            
            return {
                'status': status,
                'free_gb': round(free_gb, 2),
                'total_gb': round(total_gb, 2),
                'usage_percent': round(usage_percent, 2)
            }
        except Exception as e:
            logger.error(f"Disk space check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            status = 'healthy'
            if memory.percent > 90:
                status = 'critical'
            elif memory.percent > 75:
                status = 'warning'
            
            return {
                'status': status,
                'system_percent': memory.percent,
                'system_available_gb': round(memory.available / (1024**3), 2),
                'process_mb': round(process_memory.rss / (1024**2), 2)
            }
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_directories(self) -> Dict[str, Any]:
        """Check required directories"""
        directories = {
            'audio': AUDIO_DIR,
            'text': TEXT_DIR,
            'temp': TEMP_DIR,
            'reports': REPORTS_DIR
        }
        
        results = {}
        all_healthy = True
        
        for name, path in directories.items():
            try:
                if path.exists():
                    if path.is_dir():
                        # Count files
                        file_count = len(list(path.rglob("*")))
                        results[name] = {
                            'status': 'healthy',
                            'path': str(path),
                            'file_count': file_count
                        }
                    else:
                        results[name] = {
                            'status': 'error',
                            'path': str(path),
                            'error': 'Path is not a directory'
                        }
                        all_healthy = False
                else:
                    results[name] = {
                        'status': 'warning',
                        'path': str(path),
                        'message': 'Directory does not exist'
                    }
                    # Create missing directories
                    path.mkdir(parents=True, exist_ok=True)
                    results[name]['message'] = 'Directory created'
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'path': str(path),
                    'error': str(e)
                }
                all_healthy = False
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'directories': results
        }
    
    def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check API connectivity"""
        from src.ai_analyzer import AIAnalyzer
        
        try:
            analyzer = AIAnalyzer()
            
            # Check if configured
            if analyzer.configured:
                return {
                    'status': 'healthy',
                    'gemini_configured': True
                }
            else:
                return {
                    'status': 'warning',
                    'gemini_configured': False,
                    'message': 'Gemini API not configured'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Application metrics
            from src.pipeline_processor import PipelineProcessor
            from src.speaker_identification import SpeakerIdentificationSystem
            
            # Get pipeline stats if available
            pipeline_stats = {}
            try:
                processor = PipelineProcessor()
                pipeline_stats = processor.get_summary_stats()
            except:
                pass
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'disk_free_gb': round(disk.free / (1024**3), 2),
                    'disk_usage_percent': round((disk.used / disk.total) * 100, 2)
                },
                'process': {
                    'memory_mb': round(process_memory.rss / (1024**2), 2),
                    'cpu_percent': process_cpu,
                    'threads': process.num_threads(),
                    'open_files': process.num_fds() if hasattr(process, 'num_fds') else 'N/A'
                },
                'application': {
                    'name': APP_NAME,
                    'version': APP_VERSION,
                    'uptime_seconds': int(time.time() - self.start_time),
                    'health_status': self.health_status,
                    'pipeline_stats': pipeline_stats
                }
            }
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def log_incident(self, severity: str, message: str, details: Optional[Dict] = None):
        """Log an incident"""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        
        self.incidents.append(incident)
        
        # Log to system logger
        if severity == 'critical':
            logger.critical(f"INCIDENT: {message}")
        elif severity == 'warning':
            logger.warning(f"INCIDENT: {message}")
        else:
            logger.info(f"INCIDENT: {message}")
    
    def get_recent_incidents(self, hours: int = 24) -> list:
        """Get recent incidents"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [
            incident for incident in self.incidents
            if datetime.fromisoformat(incident['timestamp']).timestamp() > cutoff
        ]


# Global health checker instance
health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return health_checker.check_system_health()


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    return health_checker.get_metrics()


def setup_monitoring():
    """Setup monitoring and alerts"""
    logger.info("Setting up system monitoring...")
    
    # Check initial health
    health = health_checker.check_system_health()
    
    if health['status'] != 'healthy':
        health_checker.log_incident(
            'warning',
            f"System started with {health['status']} status",
            health['checks']
        )
    
    logger.info("Monitoring initialized successfully")
