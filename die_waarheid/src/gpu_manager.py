"""
GPU Detection and CUDA Optimization Module for Die Waarheid
Provides comprehensive GPU detection, memory management, and performance optimization
"""

import gc
import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# GPU configuration from environment
ENABLE_GPU = os.getenv("ENABLE_GPU", "true").lower() == "true"
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
GPU_MEMORY_GROWTH = os.getenv("GPU_MEMORY_GROWTH", "true").lower() == "true"


@dataclass
class GPUInfo:
    """GPU information container"""
    index: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # Percentage
    temperature: Optional[float] = None  # Celsius
    power_draw: Optional[float] = None   # Watts
    driver_version: Optional[str] = None


@dataclass
class CUDAInfo:
    """CUDA information container"""
    available: bool
    version: Optional[str] = None
    device_count: int = 0
    devices: List[GPUInfo] = None
    current_device: Optional[int] = None


class GPUManager:
    """
    Comprehensive GPU detection and management system
    """
    
    def __init__(self):
        """Initialize GPU manager"""
        self._cuda_info: Optional[CUDAInfo] = None
        self._torch_available = False
        self._device_cache: Dict[str, Any] = {}
        self._memory_stats: Dict[str, Any] = {}
        self._performance_stats: Dict[str, Any] = {}
        
        # Initialize GPU detection
        self._detect_gpu_capabilities()
    
    def _detect_gpu_capabilities(self) -> None:
        """Detect available GPU capabilities"""
        try:
            # Check for PyTorch CUDA support
            self._check_torch_cuda()
            
            # Check for NVIDIA GPU via nvidia-ml-py
            self._check_nvidia_ml()
            
            # Fallback to nvidia-smi
            if not self._cuda_info or not self._cuda_info.available:
                self._check_nvidia_smi()
            
            # Log detection results
            self._log_gpu_detection_results()
            
        except Exception as e:
            logger.error(f"Error during GPU detection: {str(e)}")
            self._cuda_info = CUDAInfo(available=False)
    
    def _check_torch_cuda(self) -> None:
        """Check PyTorch CUDA availability"""
        try:
            import torch
            self._torch_available = True
            
            if torch.cuda.is_available() and not FORCE_CPU:
                device_count = torch.cuda.device_count()
                devices = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory // (1024 * 1024)  # Convert to MB
                    
                    # Get current memory usage
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    memory_cached = torch.cuda.memory_reserved(i) // (1024 * 1024)
                    memory_free = memory_total - memory_cached
                    
                    device_info = GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_free=memory_free,
                        memory_used=memory_cached,
                        utilization=0.0  # PyTorch doesn't provide utilization
                    )
                    devices.append(device_info)
                
                self._cuda_info = CUDAInfo(
                    available=True,
                    version=torch.version.cuda,
                    device_count=device_count,
                    devices=devices,
                    current_device=torch.cuda.current_device()
                )
                
                logger.info(f"PyTorch CUDA detected: {device_count} device(s)")
            else:
                self._cuda_info = CUDAInfo(available=False)
                logger.info("PyTorch CUDA not available or forced CPU mode")
                
        except ImportError:
            logger.info("PyTorch not available")
            self._torch_available = False
        except Exception as e:
            logger.error(f"Error checking PyTorch CUDA: {str(e)}")
            self._torch_available = False
    
    def _check_nvidia_ml(self) -> None:
        """Check NVIDIA GPU via nvidia-ml-py"""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            devices = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temperature = None
                
                # Get power draw
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_draw = None
                
                device_info = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=memory_info.total // (1024 * 1024),
                    memory_free=memory_info.free // (1024 * 1024),
                    memory_used=memory_info.used // (1024 * 1024),
                    utilization=utilization,
                    temperature=temperature,
                    power_draw=power_draw
                )
                devices.append(device_info)
            
            # Get driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                driver_version = None
            
            # Update or create CUDA info
            if self._cuda_info:
                self._cuda_info.devices = devices
                if devices:
                    self._cuda_info.devices[0].driver_version = driver_version
            else:
                self._cuda_info = CUDAInfo(
                    available=len(devices) > 0 and not FORCE_CPU,
                    device_count=len(devices),
                    devices=devices
                )
            
            logger.info(f"NVIDIA-ML detected: {len(devices)} device(s)")
            
        except ImportError:
            logger.info("pynvml not available")
        except Exception as e:
            logger.error(f"Error checking NVIDIA-ML: {str(e)}")
    
    def _check_nvidia_smi(self) -> None:
        """Fallback GPU detection using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                devices = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            try:
                                device_info = GPUInfo(
                                    index=int(parts[0]),
                                    name=parts[1],
                                    memory_total=int(parts[2]),
                                    memory_free=int(parts[3]),
                                    memory_used=int(parts[4]),
                                    utilization=float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                                    temperature=float(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else None,
                                    power_draw=float(parts[7]) if len(parts) > 7 and parts[7] != '[Not Supported]' else None
                                )
                                devices.append(device_info)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error parsing nvidia-smi output: {e}")
                
                if devices and not self._cuda_info:
                    self._cuda_info = CUDAInfo(
                        available=not FORCE_CPU,
                        device_count=len(devices),
                        devices=devices
                    )
                elif devices and self._cuda_info:
                    # Update existing info with nvidia-smi data
                    for i, device in enumerate(devices):
                        if i < len(self._cuda_info.devices):
                            self._cuda_info.devices[i].utilization = device.utilization
                            self._cuda_info.devices[i].temperature = device.temperature
                            self._cuda_info.devices[i].power_draw = device.power_draw
                
                logger.info(f"nvidia-smi detected: {len(devices)} device(s)")
            else:
                logger.info("nvidia-smi not available or failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("nvidia-smi command not found or timed out")
        except Exception as e:
            logger.error(f"Error running nvidia-smi: {str(e)}")
    
    def _log_gpu_detection_results(self) -> None:
        """Log GPU detection results"""
        if not self._cuda_info or not self._cuda_info.available:
            logger.info("No GPU detected or GPU disabled - using CPU mode")
            return
        
        logger.info(f"GPU Detection Results:")
        logger.info(f"  CUDA Available: {self._cuda_info.available}")
        logger.info(f"  CUDA Version: {self._cuda_info.version or 'Unknown'}")
        logger.info(f"  Device Count: {self._cuda_info.device_count}")
        
        if self._cuda_info.devices:
            for device in self._cuda_info.devices:
                logger.info(f"  Device {device.index}: {device.name}")
                logger.info(f"    Memory: {device.memory_used}MB / {device.memory_total}MB")
                logger.info(f"    Utilization: {device.utilization:.1f}%")
                if device.temperature:
                    logger.info(f"    Temperature: {device.temperature}°C")
                if device.power_draw:
                    logger.info(f"    Power: {device.power_draw:.1f}W")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use"""
        return (
            ENABLE_GPU and 
            not FORCE_CPU and 
            self._cuda_info is not None and 
            self._cuda_info.available and 
            self._cuda_info.device_count > 0
        )
    
    def get_optimal_device(self) -> str:
        """
        Get optimal device for computation
        
        Returns:
            Device string ('cuda:0', 'cuda:1', or 'cpu')
        """
        if not self.is_gpu_available():
            return "cpu"
        
        if not self._cuda_info.devices:
            return "cpu"
        
        # Find device with most free memory
        best_device = min(
            self._cuda_info.devices,
            key=lambda d: d.memory_used / d.memory_total
        )
        
        return f"cuda:{best_device.index}"
    
    def get_device_info(self, device_index: Optional[int] = None) -> Optional[GPUInfo]:
        """
        Get information about specific GPU device
        
        Args:
            device_index: GPU device index (None for current/optimal device)
            
        Returns:
            GPUInfo object or None if not available
        """
        if not self.is_gpu_available() or not self._cuda_info.devices:
            return None
        
        if device_index is None:
            # Get optimal device
            optimal_device = self.get_optimal_device()
            if optimal_device == "cpu":
                return None
            device_index = int(optimal_device.split(':')[1])
        
        if 0 <= device_index < len(self._cuda_info.devices):
            return self._cuda_info.devices[device_index]
        
        return None
    
    def get_memory_info(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get memory information for GPU device
        
        Args:
            device_index: GPU device index
            
        Returns:
            Dictionary with memory information
        """
        device_info = self.get_device_info(device_index)
        if not device_info:
            return {"available": False, "device": "cpu"}
        
        memory_info = {
            "available": True,
            "device": f"cuda:{device_info.index}",
            "total_mb": device_info.memory_total,
            "free_mb": device_info.memory_free,
            "used_mb": device_info.memory_used,
            "utilization_percent": (device_info.memory_used / device_info.memory_total) * 100,
            "gpu_utilization_percent": device_info.utilization
        }
        
        # Add PyTorch-specific memory info if available
        if self._torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(device_info.index)
                    memory_info.update({
                        "torch_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                        "torch_cached_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024)
                    })
            except Exception as e:
                logger.warning(f"Error getting PyTorch memory info: {e}")
        
        return memory_info
    
    @contextmanager
    def gpu_memory_context(self, device_index: Optional[int] = None):
        """
        Context manager for GPU memory management
        
        Args:
            device_index: GPU device index
        """
        if not self.is_gpu_available():
            yield
            return
        
        device_info = self.get_device_info(device_index)
        if not device_info:
            yield
            return
        
        # Record initial memory state
        initial_memory = self.get_memory_info(device_index)
        
        try:
            yield
        finally:
            # Cleanup GPU memory
            self.cleanup_gpu_memory(device_index)
            
            # Log memory usage
            final_memory = self.get_memory_info(device_index)
            memory_delta = final_memory["used_mb"] - initial_memory["used_mb"]
            
            if abs(memory_delta) > 100:  # Log if significant change
                logger.debug(f"GPU memory change: {memory_delta:+.1f}MB on device {device_info.index}")
    
    def cleanup_gpu_memory(self, device_index: Optional[int] = None) -> None:
        """
        Clean up GPU memory
        
        Args:
            device_index: GPU device index (None for all devices)
        """
        if not self.is_gpu_available():
            return
        
        try:
            if self._torch_available:
                import torch
                if torch.cuda.is_available():
                    if device_index is not None:
                        torch.cuda.set_device(device_index)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    else:
                        # Clean all devices
                        for i in range(torch.cuda.device_count()):
                            torch.cuda.set_device(i)
                            torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("GPU memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")
    
    def optimize_model_loading(self, model_size: str = "small") -> Dict[str, Any]:
        """
        Get optimized settings for model loading
        
        Args:
            model_size: Model size for memory estimation
            
        Returns:
            Dictionary with optimization settings
        """
        settings = {
            "device": "cpu",
            "fp16": False,
            "memory_efficient": True,
            "batch_size": 1,
            "num_workers": 1
        }
        
        if not self.is_gpu_available():
            return settings
        
        device_info = self.get_device_info()
        if not device_info:
            return settings
        
        # Estimate memory requirements for different model sizes
        memory_requirements = {
            "tiny": 100,    # MB
            "base": 200,    # MB
            "small": 500,   # MB
            "medium": 1500, # MB
            "large": 3000   # MB
        }
        
        required_memory = memory_requirements.get(model_size, 1000)
        available_memory = device_info.memory_free
        
        if available_memory > required_memory * 1.5:  # 50% safety margin
            settings.update({
                "device": f"cuda:{device_info.index}",
                "fp16": True,  # Use half precision for memory efficiency
                "batch_size": min(8, available_memory // required_memory),
                "num_workers": min(4, os.cpu_count() or 1)
            })
            
            logger.info(f"GPU optimization enabled for {model_size} model on device {device_info.index}")
        else:
            logger.warning(f"Insufficient GPU memory for {model_size} model ({required_memory}MB required, {available_memory}MB available)")
        
        return settings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "gpu_available": self.is_gpu_available(),
            "torch_available": self._torch_available,
            "device_count": 0,
            "devices": [],
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count()
            }
        }
        
        if self._cuda_info:
            stats.update({
                "cuda_available": self._cuda_info.available,
                "cuda_version": self._cuda_info.version,
                "device_count": self._cuda_info.device_count,
                "current_device": self._cuda_info.current_device
            })
            
            if self._cuda_info.devices:
                stats["devices"] = [
                    {
                        "index": device.index,
                        "name": device.name,
                        "memory_total_mb": device.memory_total,
                        "memory_free_mb": device.memory_free,
                        "memory_used_mb": device.memory_used,
                        "memory_utilization_percent": (device.memory_used / device.memory_total) * 100,
                        "gpu_utilization_percent": device.utilization,
                        "temperature_celsius": device.temperature,
                        "power_draw_watts": device.power_draw
                    }
                    for device in self._cuda_info.devices
                ]
        
        return stats
    
    def monitor_gpu_usage(self, duration_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Monitor GPU usage over a period
        
        Args:
            duration_seconds: Monitoring duration
            
        Returns:
            Dictionary with monitoring results
        """
        if not self.is_gpu_available():
            return {"monitoring": False, "reason": "GPU not available"}
        
        initial_stats = self.get_performance_stats()
        time.sleep(duration_seconds)
        final_stats = self.get_performance_stats()
        
        monitoring_results = {
            "monitoring": True,
            "duration_seconds": duration_seconds,
            "initial": initial_stats,
            "final": final_stats,
            "changes": {}
        }
        
        # Calculate changes
        if initial_stats["devices"] and final_stats["devices"]:
            for i, (initial_device, final_device) in enumerate(zip(initial_stats["devices"], final_stats["devices"])):
                monitoring_results["changes"][f"device_{i}"] = {
                    "memory_change_mb": final_device["memory_used_mb"] - initial_device["memory_used_mb"],
                    "utilization_change_percent": final_device["gpu_utilization_percent"] - initial_device["gpu_utilization_percent"]
                }
        
        return monitoring_results


# Global GPU manager instance
gpu_manager = GPUManager()


def get_optimal_device() -> str:
    """Get optimal device for computation"""
    return gpu_manager.get_optimal_device()


def is_gpu_available() -> bool:
    """Check if GPU is available"""
    return gpu_manager.is_gpu_available()


def get_gpu_memory_info(device_index: Optional[int] = None) -> Dict[str, Any]:
    """Get GPU memory information"""
    return gpu_manager.get_memory_info(device_index)


def cleanup_gpu_memory(device_index: Optional[int] = None) -> None:
    """Clean up GPU memory"""
    gpu_manager.cleanup_gpu_memory(device_index)


def get_model_optimization_settings(model_size: str = "small") -> Dict[str, Any]:
    """Get optimized settings for model loading"""
    return gpu_manager.optimize_model_loading(model_size)


if __name__ == "__main__":
    # Test GPU detection and capabilities
    manager = GPUManager()
    
    print("GPU Detection Results:")
    print(f"GPU Available: {manager.is_gpu_available()}")
    print(f"Optimal Device: {manager.get_optimal_device()}")
    
    stats = manager.get_performance_stats()
    print(f"Performance Stats: {stats}")
    
    if manager.is_gpu_available():
        memory_info = manager.get_memory_info()
        print(f"Memory Info: {memory_info}")
        
        # Test memory context
        with manager.gpu_memory_context():
            print("Inside GPU memory context")
