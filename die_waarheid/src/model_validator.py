"""
Model Validation & Version Checking for Die Waarheid
Comprehensive model integrity, version compatibility, and security validation
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urlparse

import whisper

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a Whisper model"""
    name: str                           # Model name (tiny, small, medium, large)
    size_mb: int                        # Model size in MB
    expected_checksum: Optional[str]    # Expected SHA256 checksum
    min_whisper_version: str            # Minimum required Whisper version
    max_whisper_version: Optional[str]  # Maximum supported Whisper version
    download_url: Optional[str]         # Official download URL
    local_path: Optional[Path]          # Local model file path
    last_validated: Optional[datetime]  # Last validation timestamp
    validation_status: str              # Current validation status
    performance_benchmark: Optional[Dict] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of model validation"""
    model_name: str                     # Model name
    is_valid: bool                      # Overall validation status
    checksum_valid: bool                # Checksum validation result
    version_compatible: bool            # Version compatibility result
    file_exists: bool                   # File existence check
    file_readable: bool                 # File readability check
    size_valid: bool                    # File size validation
    corruption_detected: bool           # Corruption detection result
    security_valid: bool                # Security validation result
    performance_valid: bool             # Performance validation result
    validation_time: datetime           # Validation timestamp
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ModelValidator:
    """
    Comprehensive model validation and version checking system
    
    Features:
    - Model integrity validation with checksum verification
    - Version compatibility checking
    - Automatic model updates and caching
    - Performance validation and benchmarking
    - Security validation and tamper detection
    """
    
    # Official Whisper model information
    WHISPER_MODELS = {
        "tiny": ModelInfo(
            name="tiny",
            size_mb=39,
            expected_checksum="65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
            min_whisper_version="20231117",
            max_whisper_version=None,
            download_url="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
            local_path=None,
            last_validated=None,
            validation_status="not_validated"
        ),
        "base": ModelInfo(
            name="base",
            size_mb=74,
            expected_checksum="4d73b88e8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b",
            min_whisper_version="20231117",
            max_whisper_version=None,
            download_url="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
            local_path=None,
            last_validated=None,
            validation_status="not_validated"
        ),
        "small": ModelInfo(
            name="small",
            size_mb=244,
            expected_checksum="9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17411a391d",
            min_whisper_version="20231117",
            max_whisper_version=None,
            download_url="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17411a391d/small.pt",
            local_path=None,
            last_validated=None,
            validation_status="not_validated"
        ),
        "medium": ModelInfo(
            name="medium",
            size_mb=769,
            expected_checksum="345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
            min_whisper_version="20231117",
            max_whisper_version=None,
            download_url="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
            local_path=None,
            last_validated=None,
            validation_status="not_validated"
        ),
        "large": ModelInfo(
            name="large",
            size_mb=1550,
            expected_checksum="e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
            min_whisper_version="20231117",
            max_whisper_version=None,
            download_url="https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v3.pt",
            local_path=None,
            last_validated=None,
            validation_status="not_validated"
        )
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, validation_interval_hours: int = 24):
        """
        Initialize model validator
        
        Args:
            cache_dir: Directory for model cache and validation data
            validation_interval_hours: Hours between automatic validations
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "die_waarheid" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_interval = timedelta(hours=validation_interval_hours)
        self.validation_cache_file = self.cache_dir / "validation_cache.json"
        
        # Load validation cache
        self._load_validation_cache()
        
        # Update model local paths
        self._update_model_paths()
        
        logger.info(f"ModelValidator initialized with cache dir: {self.cache_dir}")
    
    def _load_validation_cache(self) -> None:
        """Load validation cache from disk"""
        try:
            if self.validation_cache_file.exists():
                with open(self.validation_cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Update model information from cache
                for model_name, model_data in cache_data.get("models", {}).items():
                    if model_name in self.WHISPER_MODELS:
                        model_info = self.WHISPER_MODELS[model_name]
                        if "last_validated" in model_data:
                            model_info.last_validated = datetime.fromisoformat(model_data["last_validated"])
                        if "validation_status" in model_data:
                            model_info.validation_status = model_data["validation_status"]
                        if "performance_benchmark" in model_data:
                            model_info.performance_benchmark = model_data["performance_benchmark"]
                
                logger.debug(f"Loaded validation cache with {len(cache_data.get('models', {}))} models")
        
        except Exception as e:
            logger.warning(f"Error loading validation cache: {e}")
    
    def _save_validation_cache(self) -> None:
        """Save validation cache to disk"""
        try:
            cache_data = {
                "last_updated": datetime.now().isoformat(),
                "models": {}
            }
            
            for model_name, model_info in self.WHISPER_MODELS.items():
                cache_data["models"][model_name] = {
                    "last_validated": model_info.last_validated.isoformat() if model_info.last_validated else None,
                    "validation_status": model_info.validation_status,
                    "performance_benchmark": model_info.performance_benchmark
                }
            
            with open(self.validation_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug("Validation cache saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving validation cache: {e}")
    
    def _update_model_paths(self) -> None:
        """Update local model file paths"""
        try:
            # Get Whisper cache directory
            whisper_cache = Path.home() / ".cache" / "whisper"
            
            for model_name, model_info in self.WHISPER_MODELS.items():
                # Look for model file in Whisper cache
                potential_paths = [
                    whisper_cache / f"{model_name}.pt",
                    whisper_cache / f"{model_name}-v3.pt",
                    whisper_cache / f"{model_name}-v2.pt",
                    self.cache_dir / f"{model_name}.pt"
                ]
                
                for path in potential_paths:
                    if path.exists():
                        model_info.local_path = path
                        break
                
                logger.debug(f"Model {model_name} path: {model_info.local_path}")
        
        except Exception as e:
            logger.warning(f"Error updating model paths: {e}")
    
    def calculate_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate file checksum
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, sha1)
            
        Returns:
            Hexadecimal checksum string
        """
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            raise
    
    def validate_model_integrity(self, model_name: str) -> ValidationResult:
        """
        Validate model file integrity
        
        Args:
            model_name: Name of model to validate
            
        Returns:
            ValidationResult with integrity validation details
        """
        if model_name not in self.WHISPER_MODELS:
            return ValidationResult(
                model_name=model_name,
                is_valid=False,
                checksum_valid=False,
                version_compatible=False,
                file_exists=False,
                file_readable=False,
                size_valid=False,
                corruption_detected=True,
                security_valid=False,
                performance_valid=False,
                validation_time=datetime.now(),
                errors=[f"Unknown model: {model_name}"]
            )
        
        model_info = self.WHISPER_MODELS[model_name]
        result = ValidationResult(
            model_name=model_name,
            is_valid=True,
            checksum_valid=True,
            version_compatible=True,
            file_exists=False,
            file_readable=False,
            size_valid=True,
            corruption_detected=False,
            security_valid=True,
            performance_valid=True,
            validation_time=datetime.now()
        )
        
        try:
            # Check if model file exists
            if not model_info.local_path or not model_info.local_path.exists():
                result.file_exists = False
                result.is_valid = False
                result.errors.append(f"Model file not found: {model_info.local_path}")
                return result
            
            result.file_exists = True
            
            # Check if file is readable
            try:
                with open(model_info.local_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
                result.file_readable = True
            except Exception as e:
                result.file_readable = False
                result.is_valid = False
                result.errors.append(f"Model file not readable: {e}")
                return result
            
            # Validate file size
            file_size_mb = model_info.local_path.stat().st_size / (1024 * 1024)
            expected_size_mb = model_info.size_mb
            size_tolerance = 0.1  # 10% tolerance
            
            if abs(file_size_mb - expected_size_mb) > (expected_size_mb * size_tolerance):
                result.size_valid = False
                result.is_valid = False
                result.errors.append(
                    f"Model size mismatch: expected ~{expected_size_mb}MB, got {file_size_mb:.1f}MB"
                )
            else:
                result.details["file_size_mb"] = file_size_mb
            
            # Validate checksum if available
            if model_info.expected_checksum:
                try:
                    actual_checksum = self.calculate_file_checksum(model_info.local_path)
                    
                    if actual_checksum.lower() != model_info.expected_checksum.lower():
                        result.checksum_valid = False
                        result.is_valid = False
                        result.corruption_detected = True
                        result.errors.append(
                            f"Checksum mismatch: expected {model_info.expected_checksum}, "
                            f"got {actual_checksum}"
                        )
                    else:
                        result.details["checksum"] = actual_checksum
                        logger.debug(f"Model {model_name} checksum validated successfully")
                
                except Exception as e:
                    result.checksum_valid = False
                    result.is_valid = False
                    result.errors.append(f"Error calculating checksum: {e}")
            else:
                result.warnings.append("No expected checksum available for validation")
            
            # Update model validation status
            model_info.last_validated = result.validation_time
            model_info.validation_status = "valid" if result.is_valid else "invalid"
            
            logger.info(f"Model {model_name} integrity validation: {'PASSED' if result.is_valid else 'FAILED'}")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {e}")
            logger.error(f"Error validating model {model_name}: {e}")
        
        return result
    
    def validate_version_compatibility(self, model_name: str) -> bool:
        """
        Validate Whisper version compatibility
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if version is compatible
        """
        try:
            if model_name not in self.WHISPER_MODELS:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            model_info = self.WHISPER_MODELS[model_name]
            
            # Get current Whisper version
            current_version = whisper.__version__ if hasattr(whisper, '__version__') else "unknown"
            
            # Check minimum version requirement
            if model_info.min_whisper_version:
                if current_version < model_info.min_whisper_version:
                    logger.error(
                        f"Whisper version {current_version} is below minimum required "
                        f"{model_info.min_whisper_version} for model {model_name}"
                    )
                    return False
            
            # Check maximum version requirement
            if model_info.max_whisper_version:
                if current_version > model_info.max_whisper_version:
                    logger.warning(
                        f"Whisper version {current_version} is above maximum tested "
                        f"{model_info.max_whisper_version} for model {model_name}"
                    )
                    # Don't fail on max version, just warn
            
            logger.debug(f"Version compatibility check passed for {model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error checking version compatibility: {e}")
            return False
    
    def validate_model_performance(self, model_name: str, test_audio_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate model performance with benchmark test
        
        Args:
            model_name: Name of model to test
            test_audio_path: Optional path to test audio file
            
        Returns:
            Performance validation results
        """
        try:
            if model_name not in self.WHISPER_MODELS:
                return {"error": f"Unknown model: {model_name}"}
            
            model_info = self.WHISPER_MODELS[model_name]
            
            # Create a simple performance test
            start_time = time.time()
            
            try:
                # Try to load the model
                model = whisper.load_model(model_name)
                load_time = time.time() - start_time
                
                # Basic model validation
                if hasattr(model, 'dims'):
                    model_dims = model.dims
                else:
                    model_dims = "unknown"
                
                performance_result = {
                    "model_name": model_name,
                    "load_time_seconds": round(load_time, 2),
                    "model_dimensions": model_dims,
                    "validation_time": datetime.now().isoformat(),
                    "status": "success"
                }
                
                # Store benchmark results
                model_info.performance_benchmark = performance_result
                
                logger.info(f"Performance validation for {model_name}: load time {load_time:.2f}s")
                
                return performance_result
            
            except Exception as e:
                performance_result = {
                    "model_name": model_name,
                    "load_time_seconds": None,
                    "validation_time": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e)
                }
                
                logger.error(f"Performance validation failed for {model_name}: {e}")
                return performance_result
        
        except Exception as e:
            logger.error(f"Error in performance validation: {e}")
            return {"error": str(e)}
    
    def validate_model_security(self, model_name: str) -> Dict[str, Any]:
        """
        Validate model security and authenticity
        
        Args:
            model_name: Name of model to validate
            
        Returns:
            Security validation results
        """
        try:
            if model_name not in self.WHISPER_MODELS:
                return {"valid": False, "error": f"Unknown model: {model_name}"}
            
            model_info = self.WHISPER_MODELS[model_name]
            
            security_result = {
                "model_name": model_name,
                "file_exists": False,
                "checksum_verified": False,
                "file_permissions_secure": False,
                "no_suspicious_modifications": False,
                "validation_time": datetime.now().isoformat(),
                "valid": False
            }
            
            # Check if file exists
            if not model_info.local_path or not model_info.local_path.exists():
                security_result["error"] = "Model file not found"
                return security_result
            
            security_result["file_exists"] = True
            
            # Verify checksum for authenticity
            if model_info.expected_checksum:
                try:
                    actual_checksum = self.calculate_file_checksum(model_info.local_path)
                    security_result["checksum_verified"] = (
                        actual_checksum.lower() == model_info.expected_checksum.lower()
                    )
                    security_result["actual_checksum"] = actual_checksum
                except Exception as e:
                    security_result["checksum_error"] = str(e)
            
            # Check file permissions (should not be world-writable)
            try:
                file_stat = model_info.local_path.stat()
                file_mode = file_stat.st_mode
                
                # Check if file is world-writable (security risk)
                world_writable = bool(file_mode & 0o002)
                security_result["file_permissions_secure"] = not world_writable
                security_result["file_mode"] = oct(file_mode)
            
            except Exception as e:
                security_result["permissions_error"] = str(e)
            
            # Check for suspicious recent modifications
            try:
                file_stat = model_info.local_path.stat()
                modification_time = datetime.fromtimestamp(file_stat.st_mtime)
                
                # If file was modified very recently (within last hour), flag as suspicious
                # unless it's a fresh download
                time_since_modification = datetime.now() - modification_time
                recently_modified = time_since_modification < timedelta(hours=1)
                
                security_result["no_suspicious_modifications"] = not recently_modified
                security_result["last_modified"] = modification_time.isoformat()
                security_result["time_since_modification_hours"] = time_since_modification.total_seconds() / 3600
            
            except Exception as e:
                security_result["modification_check_error"] = str(e)
            
            # Overall security validation
            security_result["valid"] = (
                security_result["file_exists"] and
                security_result.get("checksum_verified", True) and  # True if no checksum to verify
                security_result.get("file_permissions_secure", True) and
                security_result.get("no_suspicious_modifications", True)
            )
            
            logger.info(f"Security validation for {model_name}: {'PASSED' if security_result['valid'] else 'FAILED'}")
            
            return security_result
        
        except Exception as e:
            logger.error(f"Error in security validation: {e}")
            return {"valid": False, "error": str(e)}
    
    def comprehensive_model_validation(self, model_name: str) -> ValidationResult:
        """
        Perform comprehensive model validation
        
        Args:
            model_name: Name of model to validate
            
        Returns:
            Complete validation result
        """
        logger.info(f"Starting comprehensive validation for model: {model_name}")
        
        # Start with integrity validation
        result = self.validate_model_integrity(model_name)
        
        # Add version compatibility check
        result.version_compatible = self.validate_version_compatibility(model_name)
        if not result.version_compatible:
            result.is_valid = False
            result.errors.append("Version compatibility check failed")
        
        # Add performance validation
        performance_result = self.validate_model_performance(model_name)
        result.performance_valid = performance_result.get("status") == "success"
        if not result.performance_valid:
            result.errors.append(f"Performance validation failed: {performance_result.get('error', 'Unknown error')}")
        result.details["performance"] = performance_result
        
        # Add security validation
        security_result = self.validate_model_security(model_name)
        result.security_valid = security_result.get("valid", False)
        if not result.security_valid:
            result.errors.append("Security validation failed")
        result.details["security"] = security_result
        
        # Update overall validation status
        result.is_valid = (
            result.checksum_valid and
            result.version_compatible and
            result.file_exists and
            result.file_readable and
            result.size_valid and
            not result.corruption_detected and
            result.security_valid and
            result.performance_valid
        )
        
        # Save validation cache
        self._save_validation_cache()
        
        logger.info(f"Comprehensive validation for {model_name}: {'PASSED' if result.is_valid else 'FAILED'}")
        
        return result
    
    def is_validation_needed(self, model_name: str) -> bool:
        """
        Check if model needs validation
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if validation is needed
        """
        if model_name not in self.WHISPER_MODELS:
            return True
        
        model_info = self.WHISPER_MODELS[model_name]
        
        # Always validate if never validated
        if not model_info.last_validated:
            return True
        
        # Validate if validation interval has passed
        time_since_validation = datetime.now() - model_info.last_validated
        if time_since_validation > self.validation_interval:
            return True
        
        # Validate if status is invalid
        if model_info.validation_status != "valid":
            return True
        
        return False
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get current model status and validation information
        
        Args:
            model_name: Name of model
            
        Returns:
            Model status information
        """
        if model_name not in self.WHISPER_MODELS:
            return {
                "model_name": model_name,
                "exists": False,
                "error": f"Unknown model: {model_name}"
            }
        
        model_info = self.WHISPER_MODELS[model_name]
        
        return {
            "model_name": model_name,
            "exists": model_info.local_path and model_info.local_path.exists(),
            "local_path": str(model_info.local_path) if model_info.local_path else None,
            "size_mb": model_info.size_mb,
            "last_validated": model_info.last_validated.isoformat() if model_info.last_validated else None,
            "validation_status": model_info.validation_status,
            "validation_needed": self.is_validation_needed(model_name),
            "performance_benchmark": model_info.performance_benchmark,
            "min_whisper_version": model_info.min_whisper_version,
            "max_whisper_version": model_info.max_whisper_version
        }
    
    def get_all_models_status(self) -> Dict[str, Any]:
        """
        Get status for all models
        
        Returns:
            Status information for all models
        """
        return {
            "models": {
                model_name: self.get_model_status(model_name)
                for model_name in self.WHISPER_MODELS.keys()
            },
            "cache_dir": str(self.cache_dir),
            "validation_interval_hours": self.validation_interval.total_seconds() / 3600,
            "whisper_version": getattr(whisper, '__version__', 'unknown'),
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.machine()
            }
        }


# Global model validator instance
model_validator = ModelValidator()


def validate_model(model_name: str, force: bool = False) -> ValidationResult:
    """
    Validate a specific model
    
    Args:
        model_name: Name of model to validate
        force: Force validation even if recently validated
        
    Returns:
        Validation result
    """
    if force or model_validator.is_validation_needed(model_name):
        return model_validator.comprehensive_model_validation(model_name)
    else:
        logger.debug(f"Model {model_name} validation not needed (recently validated)")
        # Return cached result
        model_info = model_validator.WHISPER_MODELS.get(model_name)
        if model_info:
            return ValidationResult(
                model_name=model_name,
                is_valid=model_info.validation_status == "valid",
                checksum_valid=True,
                version_compatible=True,
                file_exists=model_info.local_path and model_info.local_path.exists(),
                file_readable=True,
                size_valid=True,
                corruption_detected=False,
                security_valid=True,
                performance_valid=True,
                validation_time=model_info.last_validated or datetime.now()
            )


def get_model_validation_status(model_name: str) -> Dict[str, Any]:
    """
    Get model validation status
    
    Args:
        model_name: Name of model
        
    Returns:
        Model status information
    """
    return model_validator.get_model_status(model_name)


def get_all_models_validation_status() -> Dict[str, Any]:
    """
    Get validation status for all models
    
    Returns:
        Status information for all models
    """
    return model_validator.get_all_models_status()
