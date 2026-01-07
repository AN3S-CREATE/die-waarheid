"""
Health Monitoring & System Status for Die Waarheid
Comprehensive system health checks, metrics collection, and status reporting
"""

import logging
import os
import platform
import psutil
import shutil
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float                  # CPU utilization percentage
    memory_percent: float               # Memory utilization percentage
    memory_available_mb: float          # Available memory in MB
    memory_used_mb: float               # Used memory in MB
    memory_total_mb: float              # Total memory in MB
    disk_usage_percent: float           # Disk usage percentage
    disk_free_gb: float                 # Free disk space in GB
    disk_total_gb: float                # Total disk space in GB
    network_bytes_sent: int             # Network bytes sent
    network_bytes_recv: int             # Network bytes received
    load_average: Optional[List[float]] # System load average (1, 5, 15 min)
    process_count: int                  # Number of running processes
    uptime_seconds: float               # System uptime in seconds


@dataclass
class ComponentHealth:
    """Health status of individual components"""
    name: str                           # Component name
    status: str                         # Health status (healthy, warning, critical, unknown)
    last_check: datetime                # Last health check timestamp
    response_time_ms: Optional[float]   # Response time in milliseconds
    error_message: Optional[str]        # Error message if unhealthy
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthThresholds:
    """Health monitoring thresholds"""
    cpu_warning: float = 80.0           # CPU warning threshold (%)
    cpu_critical: float = 95.0          # CPU critical threshold (%)
    memory_warning: float = 85.0        # Memory warning threshold (%)
    memory_critical: float = 95.0       # Memory critical threshold (%)
    disk_warning: float = 85.0          # Disk warning threshold (%)
    disk_critical: float = 95.0         # Disk critical threshold (%)
    response_time_warning: float = 1000.0  # Response time warning (ms)
    response_time_critical: float = 5000.0  # Response time critical (ms)
    load_average_warning: float = 2.0   # Load average warning
    load_average_critical: float = 5.0  # Load average critical


class HealthMonitor:
    """
    Comprehensive health monitoring and system status reporting
    
    Features:
    - System metrics collection (CPU, memory, disk, network)
    - Component health tracking
    - Performance monitoring and alerting
    - Health history and trends
    - Configurable thresholds and alerts
    """
    
    def __init__(self, 
                 metrics_history_size: int = 100,
                 health_check_interval: int = 30,
                 thresholds: Optional[HealthThresholds] = None):
        """
        Initialize health monitor
        
        Args:
            metrics_history_size: Number of metrics entries to keep in history
            health_check_interval: Interval between health checks in seconds
            thresholds: Health monitoring thresholds
        """
        self.metrics_history_size = metrics_history_size
        self.health_check_interval = health_check_interval
        self.thresholds = thresholds or HealthThresholds()
        
        # Health monitoring state
        self.metrics_history: List[SystemMetrics] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.last_health_check = datetime.now()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Network baseline
        self._network_baseline = self._get_network_baseline()
        
        logger.info(f"HealthMonitor initialized with {metrics_history_size} metrics history")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information"""
        try:
            return {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error collecting system info: {e}")
            return {"error": str(e)}
    
    def _get_network_baseline(self) -> Dict[str, int]:
        """Get network baseline for delta calculations"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.warning(f"Error getting network baseline: {e}")
            return {"bytes_sent": 0, "bytes_recv": 0, "timestamp": time.time()}
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics
        
        Returns:
            SystemMetrics with current system performance data
        """
        try:
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # Disk metrics (for root filesystem)
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / (1024**3)
            disk_total_gb = disk_usage.total / (1024**3)
            
            # Network metrics (delta from baseline)
            net_io = psutil.net_io_counters()
            current_time = time.time()
            time_delta = current_time - self._network_baseline["timestamp"]
            
            if time_delta > 0:
                network_bytes_sent = int((net_io.bytes_sent - self._network_baseline["bytes_sent"]) / time_delta)
                network_bytes_recv = int((net_io.bytes_recv - self._network_baseline["bytes_recv"]) / time_delta)
            else:
                network_bytes_sent = 0
                network_bytes_recv = 0
            
            # Update network baseline
            self._network_baseline = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "timestamp": current_time
            }
            
            # Load average (Unix-like systems only)
            load_average = None
            try:
                if hasattr(os, 'getloadavg'):
                    load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                pass
            
            # Process count
            process_count = len(psutil.pids())
            
            # System uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                disk_total_gb=disk_total_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
            # Add to history
            with self._lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.metrics_history_size:
                    self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                disk_total_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                load_average=None,
                process_count=0,
                uptime_seconds=0.0
            )
    
    def check_component_health(self, component_name: str, check_function, timeout: float = 5.0) -> ComponentHealth:
        """
        Check health of a specific component
        
        Args:
            component_name: Name of the component
            check_function: Function to call for health check
            timeout: Timeout for health check in seconds
            
        Returns:
            ComponentHealth with check results
        """
        start_time = time.time()
        
        try:
            # Execute health check with timeout
            result = check_function()
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine status based on response time and result
            if response_time_ms > self.thresholds.response_time_critical:
                status = "critical"
                error_message = f"Response time {response_time_ms:.1f}ms exceeds critical threshold"
            elif response_time_ms > self.thresholds.response_time_warning:
                status = "warning"
                error_message = f"Response time {response_time_ms:.1f}ms exceeds warning threshold"
            elif result is False:
                status = "critical"
                error_message = "Health check returned False"
            elif isinstance(result, dict) and result.get("status") == "error":
                status = "critical"
                error_message = result.get("message", "Health check reported error")
            else:
                status = "healthy"
                error_message = None
            
            component_health = ComponentHealth(
                name=component_name,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time_ms,
                error_message=error_message,
                details=result if isinstance(result, dict) else {"result": result}
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            component_health = ComponentHealth(
                name=component_name,
                status="critical",
                last_check=datetime.now(),
                response_time_ms=response_time_ms,
                error_message=f"Health check failed: {str(e)}",
                details={"exception": str(e)}
            )
        
        # Store component health
        with self._lock:
            self.component_health[component_name] = component_health
        
        return component_health
    
    def evaluate_system_health(self, metrics: SystemMetrics) -> str:
        """
        Evaluate overall system health based on metrics
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Overall health status (healthy, warning, critical)
        """
        critical_conditions = []
        warning_conditions = []
        
        # Check CPU usage
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            critical_conditions.append(f"CPU usage {metrics.cpu_percent:.1f}% >= {self.thresholds.cpu_critical}%")
        elif metrics.cpu_percent >= self.thresholds.cpu_warning:
            warning_conditions.append(f"CPU usage {metrics.cpu_percent:.1f}% >= {self.thresholds.cpu_warning}%")
        
        # Check memory usage
        if metrics.memory_percent >= self.thresholds.memory_critical:
            critical_conditions.append(f"Memory usage {metrics.memory_percent:.1f}% >= {self.thresholds.memory_critical}%")
        elif metrics.memory_percent >= self.thresholds.memory_warning:
            warning_conditions.append(f"Memory usage {metrics.memory_percent:.1f}% >= {self.thresholds.memory_warning}%")
        
        # Check disk usage
        if metrics.disk_usage_percent >= self.thresholds.disk_critical:
            critical_conditions.append(f"Disk usage {metrics.disk_usage_percent:.1f}% >= {self.thresholds.disk_critical}%")
        elif metrics.disk_usage_percent >= self.thresholds.disk_warning:
            warning_conditions.append(f"Disk usage {metrics.disk_usage_percent:.1f}% >= {self.thresholds.disk_warning}%")
        
        # Check load average (if available)
        if metrics.load_average and len(metrics.load_average) > 0:
            load_1min = metrics.load_average[0]
            if load_1min >= self.thresholds.load_average_critical:
                critical_conditions.append(f"Load average {load_1min:.2f} >= {self.thresholds.load_average_critical}")
            elif load_1min >= self.thresholds.load_average_warning:
                warning_conditions.append(f"Load average {load_1min:.2f} >= {self.thresholds.load_average_warning}")
        
        # Determine overall status
        if critical_conditions:
            return "critical"
        elif warning_conditions:
            return "warning"
        else:
            return "healthy"
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary
        
        Returns:
            Dictionary with complete health status
        """
        try:
            # Collect current metrics
            current_metrics = self.collect_system_metrics()
            
            # Evaluate system health
            system_status = self.evaluate_system_health(current_metrics)
            
            # Component health summary
            component_summary = {}
            component_statuses = {"healthy": 0, "warning": 0, "critical": 0, "unknown": 0}
            
            with self._lock:
                for name, health in self.component_health.items():
                    component_summary[name] = {
                        "status": health.status,
                        "last_check": health.last_check.isoformat(),
                        "response_time_ms": health.response_time_ms,
                        "error_message": health.error_message
                    }
                    component_statuses[health.status] = component_statuses.get(health.status, 0) + 1
            
            # Calculate metrics trends (if we have history)
            trends = self._calculate_trends()
            
            health_summary = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": system_status,
                "system_metrics": {
                    "cpu_percent": current_metrics.cpu_percent,
                    "memory_percent": current_metrics.memory_percent,
                    "memory_available_mb": current_metrics.memory_available_mb,
                    "memory_used_mb": current_metrics.memory_used_mb,
                    "memory_total_mb": current_metrics.memory_total_mb,
                    "disk_usage_percent": current_metrics.disk_usage_percent,
                    "disk_free_gb": current_metrics.disk_free_gb,
                    "disk_total_gb": current_metrics.disk_total_gb,
                    "network_bytes_sent_per_sec": current_metrics.network_bytes_sent,
                    "network_bytes_recv_per_sec": current_metrics.network_bytes_recv,
                    "load_average": current_metrics.load_average,
                    "process_count": current_metrics.process_count,
                    "uptime_seconds": current_metrics.uptime_seconds,
                    "uptime_human": self._format_uptime(current_metrics.uptime_seconds)
                },
                "component_health": component_summary,
                "component_summary": component_statuses,
                "trends": trends,
                "thresholds": {
                    "cpu_warning": self.thresholds.cpu_warning,
                    "cpu_critical": self.thresholds.cpu_critical,
                    "memory_warning": self.thresholds.memory_warning,
                    "memory_critical": self.thresholds.memory_critical,
                    "disk_warning": self.thresholds.disk_warning,
                    "disk_critical": self.thresholds.disk_critical,
                    "response_time_warning": self.thresholds.response_time_warning,
                    "response_time_critical": self.thresholds.response_time_critical
                },
                "system_info": self.system_info,
                "monitoring": {
                    "active": self.monitoring_active,
                    "check_interval": self.health_check_interval,
                    "metrics_history_size": len(self.metrics_history),
                    "last_check": self.last_health_check.isoformat()
                }
            }
            
            return health_summary
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "unknown",
                "error": str(e)
            }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from metrics history"""
        try:
            with self._lock:
                if len(self.metrics_history) < 2:
                    return {"available": False, "reason": "Insufficient data"}
                
                # Get recent metrics for trend calculation
                recent_metrics = self.metrics_history[-10:]  # Last 10 entries
                
                # Calculate averages and trends
                cpu_values = [m.cpu_percent for m in recent_metrics]
                memory_values = [m.memory_percent for m in recent_metrics]
                disk_values = [m.disk_usage_percent for m in recent_metrics]
                
                trends = {
                    "available": True,
                    "period_minutes": len(recent_metrics) * (self.health_check_interval / 60),
                    "cpu": {
                        "average": sum(cpu_values) / len(cpu_values),
                        "min": min(cpu_values),
                        "max": max(cpu_values),
                        "trend": "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
                    },
                    "memory": {
                        "average": sum(memory_values) / len(memory_values),
                        "min": min(memory_values),
                        "max": max(memory_values),
                        "trend": "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
                    },
                    "disk": {
                        "average": sum(disk_values) / len(disk_values),
                        "min": min(disk_values),
                        "max": max(disk_values),
                        "trend": "increasing" if disk_values[-1] > disk_values[0] else "decreasing"
                    }
                }
                
                return trends
                
        except Exception as e:
            logger.warning(f"Error calculating trends: {e}")
            return {"available": False, "error": str(e)}
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format"""
        try:
            uptime_delta = timedelta(seconds=int(uptime_seconds))
            days = uptime_delta.days
            hours, remainder = divmod(uptime_delta.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except Exception:
            return f"{int(uptime_seconds)}s"
    
    def start_monitoring(self) -> None:
        """Start background health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                self.collect_system_metrics()
                self.last_health_check = datetime.now()
                
                # Sleep for the specified interval
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.health_check_interval)
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get metrics history
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of historical metrics
        """
        with self._lock:
            history = self.metrics_history.copy()
        
        if limit:
            history = history[-limit:]
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "memory_available_mb": m.memory_available_mb,
                "memory_used_mb": m.memory_used_mb,
                "disk_usage_percent": m.disk_usage_percent,
                "disk_free_gb": m.disk_free_gb,
                "network_bytes_sent": m.network_bytes_sent,
                "network_bytes_recv": m.network_bytes_recv,
                "load_average": m.load_average,
                "process_count": m.process_count,
                "uptime_seconds": m.uptime_seconds
            }
            for m in history
        ]
    
    def export_health_report(self, filepath: Optional[Path] = None) -> Path:
        """
        Export comprehensive health report to JSON file
        
        Args:
            filepath: Optional path for the report file
            
        Returns:
            Path to the exported report file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"health_report_{timestamp}.json")
        
        try:
            health_summary = self.get_health_summary()
            metrics_history = self.get_metrics_history()
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "health_summary": health_summary,
                "metrics_history": metrics_history,
                "system_info": self.system_info
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Health report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting health report: {e}")
            raise


# Global health monitor instance
health_monitor = HealthMonitor()


def get_health_summary() -> Dict[str, Any]:
    """
    Get comprehensive health summary
    
    Returns:
        Complete health status information
    """
    return health_monitor.get_health_summary()


def check_component_health(component_name: str, check_function, timeout: float = 5.0) -> ComponentHealth:
    """
    Check health of a specific component
    
    Args:
        component_name: Name of the component
        check_function: Function to call for health check
        timeout: Timeout for health check in seconds
        
    Returns:
        ComponentHealth with check results
    """
    return health_monitor.check_component_health(component_name, check_function, timeout)


def start_health_monitoring() -> None:
    """Start background health monitoring"""
    health_monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop background health monitoring"""
    health_monitor.stop_monitoring()


def get_metrics_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get metrics history
    
    Args:
        limit: Maximum number of entries to return
        
    Returns:
        List of historical metrics
    """
    return health_monitor.get_metrics_history(limit)
