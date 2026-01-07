"""
Advanced Security Module for Die Waarheid
Provides comprehensive input sanitization, validation, and security headers
"""

import html
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import quote, unquote

import bleach
from fastapi import HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Security configuration
ENABLE_ADVANCED_SECURITY = os.getenv("ENABLE_ADVANCED_SECURITY", "true").lower() == "true"
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "50000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

# Dangerous patterns for injection detection
INJECTION_PATTERNS = [
    # SQL injection patterns
    r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
    r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
    r"(?i)(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
    
    # Command injection patterns
    r"(?i)(;|\||&|`|\$\(|\${|<\(|>\()",
    r"(?i)(rm\s+-rf|del\s+/|format\s+c:)",
    r"(?i)(wget|curl|nc\s+|netcat)",
    
    # Script injection patterns
    r"(?i)(<script|</script>|javascript:|vbscript:)",
    r"(?i)(eval\s*\(|exec\s*\(|system\s*\()",
    r"(?i)(document\.|window\.|alert\s*\()",
    
    # Path traversal patterns
    r"(?i)(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
    r"(?i)(/etc/passwd|/etc/shadow|c:\\windows\\system32)",
    
    # Template injection patterns
    r"(?i)(\{\{.*\}\}|\{%.*%\}|\${.*})",
    r"(?i)(__import__|getattr|setattr|delattr)",
]

# Compiled regex patterns for performance
COMPILED_PATTERNS = [re.compile(pattern) for pattern in INJECTION_PATTERNS]

# Allowed HTML tags for content sanitization
ALLOWED_HTML_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'i', 'b',
    'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
]

ALLOWED_HTML_ATTRIBUTES = {
    '*': ['class', 'id'],
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'width', 'height']
}

# Rate limiting storage
_rate_limit_storage: Dict[str, List[float]] = {}


class SecurityValidator:
    """Advanced security validation and sanitization"""
    
    def __init__(self):
        """Initialize security validator"""
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: List[str] = []
        self.security_events: List[Dict[str, Any]] = []
        
    def sanitize_text_input(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        allow_html: bool = False,
        strict_mode: bool = True
    ) -> str:
        """
        Comprehensive text input sanitization
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow safe HTML tags
            strict_mode: Enable strict sanitization
            
        Returns:
            Sanitized text
            
        Raises:
            HTTPException: If input contains dangerous patterns
        """
        if not isinstance(text, str):
            return ""
        
        # Length validation
        max_len = max_length or MAX_INPUT_LENGTH
        if len(text) > max_len:
            raise HTTPException(
                status_code=413,
                detail=f"Input too long. Maximum length: {max_len}"
            )
        
        # Injection detection
        if strict_mode and self._detect_injection_attempts(text):
            self._log_security_event("injection_attempt", {"input": text[:100]})
            raise HTTPException(
                status_code=400,
                detail="Potentially malicious input detected"
            )
        
        # HTML sanitization
        if allow_html:
            text = bleach.clean(
                text,
                tags=ALLOWED_HTML_TAGS,
                attributes=ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape HTML entities
            text = html.escape(text)
        
        # Remove null bytes and control characters
        text = self._remove_control_characters(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            return "unknown_file"
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        # Ensure not empty
        if not filename or filename.isspace():
            filename = f"file_{secrets.token_hex(4)}"
        
        return filename
    
    def validate_file_upload(
        self, 
        file_content: bytes, 
        filename: str,
        allowed_extensions: Optional[List[str]] = None,
        max_size: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validate uploaded file for security
        
        Args:
            file_content: File content bytes
            filename: Original filename
            allowed_extensions: List of allowed file extensions
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Size validation
            if max_size and len(file_content) > max_size:
                return False, f"File too large. Maximum size: {max_size} bytes"
            
            # Extension validation
            if allowed_extensions:
                file_ext = Path(filename).suffix.lower()
                if file_ext not in allowed_extensions:
                    return False, f"File type not allowed. Allowed: {allowed_extensions}"
            
            # Magic number validation for common file types
            if not self._validate_file_magic_number(file_content, filename):
                return False, "File content doesn't match extension"
            
            # Scan for embedded scripts or malicious content
            if self._scan_file_content(file_content):
                self._log_security_event("malicious_file_upload", {"filename": filename})
                return False, "File contains potentially malicious content"
            
            return True, "File validation passed"
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            return False, "File validation failed"
    
    def check_rate_limit(
        self, 
        client_ip: str, 
        endpoint: str,
        requests_limit: Optional[int] = None,
        window_seconds: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limiting for client
        
        Args:
            client_ip: Client IP address
            endpoint: API endpoint
            requests_limit: Maximum requests per window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not ENABLE_ADVANCED_SECURITY:
            return True, {}
        
        limit = requests_limit or RATE_LIMIT_REQUESTS
        window = window_seconds or RATE_LIMIT_WINDOW
        
        key = f"{client_ip}:{endpoint}"
        current_time = time.time()
        
        # Initialize or clean old entries
        if key not in _rate_limit_storage:
            _rate_limit_storage[key] = []
        
        # Remove old requests outside the window
        _rate_limit_storage[key] = [
            req_time for req_time in _rate_limit_storage[key]
            if current_time - req_time < window
        ]
        
        # Check if limit exceeded
        if len(_rate_limit_storage[key]) >= limit:
            self._log_security_event("rate_limit_exceeded", {
                "client_ip": client_ip,
                "endpoint": endpoint,
                "requests": len(_rate_limit_storage[key])
            })
            
            return False, {
                "requests_made": len(_rate_limit_storage[key]),
                "requests_limit": limit,
                "window_seconds": window,
                "retry_after": window
            }
        
        # Add current request
        _rate_limit_storage[key].append(current_time)
        
        return True, {
            "requests_made": len(_rate_limit_storage[key]),
            "requests_limit": limit,
            "window_seconds": window
        }
    
    def generate_security_headers(self) -> Dict[str, str]:
        """
        Generate security headers for HTTP responses
        
        Returns:
            Dictionary of security headers
        """
        return {
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS enforcement
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            ),
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            )
        }
    
    def get_security_report(self) -> Dict[str, Any]:
        """
        Get comprehensive security report
        
        Returns:
            Security statistics and events
        """
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "security_events_count": len(self.security_events),
            "recent_events": self.security_events[-10:],  # Last 10 events
            "rate_limit_entries": len(_rate_limit_storage),
            "advanced_security_enabled": ENABLE_ADVANCED_SECURITY,
            "configuration": {
                "max_input_length": MAX_INPUT_LENGTH,
                "rate_limit_requests": RATE_LIMIT_REQUESTS,
                "rate_limit_window": RATE_LIMIT_WINDOW
            }
        }
    
    def _detect_injection_attempts(self, text: str) -> bool:
        """Detect potential injection attempts"""
        text_lower = text.lower()
        
        for pattern in COMPILED_PATTERNS:
            if pattern.search(text_lower):
                return True
        
        return False
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters from text"""
        # Keep only printable characters and common whitespace
        return ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    def _validate_file_magic_number(self, content: bytes, filename: str) -> bool:
        """Validate file magic number matches extension"""
        if len(content) < 4:
            return True  # Too small to validate
        
        ext = Path(filename).suffix.lower()
        magic = content[:4]
        
        # Common file type magic numbers
        magic_numbers = {
            '.pdf': [b'%PDF'],
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.png': [b'\x89PNG'],
            '.gif': [b'GIF8'],
            '.wav': [b'RIFF'],
            '.mp3': [b'ID3', b'\xff\xfb'],
            '.mp4': [b'ftyp'],
            '.zip': [b'PK\x03\x04'],
            '.docx': [b'PK\x03\x04'],  # Office docs are ZIP-based
            '.xlsx': [b'PK\x03\x04'],
        }
        
        if ext in magic_numbers:
            expected_magic = magic_numbers[ext]
            return any(magic.startswith(expected) for expected in expected_magic)
        
        return True  # Unknown extension, allow
    
    def _scan_file_content(self, content: bytes) -> bool:
        """Scan file content for malicious patterns"""
        try:
            # Convert to string for pattern matching (ignore decode errors)
            text_content = content.decode('utf-8', errors='ignore')
            
            # Check for script tags and dangerous patterns
            dangerous_patterns = [
                b'<script',
                b'javascript:',
                b'vbscript:',
                b'data:text/html',
                b'<?php',
                b'<%',
                b'#!/bin/',
                b'#!/usr/bin/',
            ]
            
            content_lower = content.lower()
            for pattern in dangerous_patterns:
                if pattern in content_lower:
                    return True
            
            return False
            
        except Exception:
            # If we can't scan, err on the side of caution
            return False
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        logger.warning(f"Security event: {event_type} - {details}")


# Global security validator instance
security_validator = SecurityValidator()


def add_security_headers(response: Response) -> Response:
    """
    Add security headers to response
    
    Args:
        response: FastAPI response object
        
    Returns:
        Response with security headers added
    """
    if ENABLE_ADVANCED_SECURITY:
        headers = security_validator.generate_security_headers()
        for key, value in headers.items():
            response.headers[key] = value
    
    return response


def validate_request_security(request: Request) -> None:
    """
    Validate request for security issues
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If security validation fails
    """
    if not ENABLE_ADVANCED_SECURITY:
        return
    
    client_ip = request.client.host if request.client else "unknown"
    
    # Check if IP is blocked
    if client_ip in security_validator.blocked_ips:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    # Rate limiting check
    endpoint = str(request.url.path)
    is_allowed, rate_info = security_validator.check_rate_limit(client_ip, endpoint)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(rate_info.get("retry_after", 3600))}
        )


# Convenience functions for common operations
def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input text"""
    return security_validator.sanitize_text_input(
        text, 
        max_length=max_length, 
        allow_html=False, 
        strict_mode=True
    )


def sanitize_html_content(html_content: str, max_length: int = 50000) -> str:
    """Sanitize HTML content allowing safe tags"""
    return security_validator.sanitize_text_input(
        html_content,
        max_length=max_length,
        allow_html=True,
        strict_mode=True
    )


def validate_file_security(
    content: bytes, 
    filename: str,
    allowed_types: Optional[List[str]] = None
) -> None:
    """
    Validate file for security (raises HTTPException if invalid)
    
    Args:
        content: File content
        filename: Original filename
        allowed_types: List of allowed file extensions
        
    Raises:
        HTTPException: If file validation fails
    """
    is_valid, error_msg = security_validator.validate_file_upload(
        content, filename, allowed_types
    )
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)


if __name__ == "__main__":
    # Test security functions
    validator = SecurityValidator()
    
    # Test input sanitization
    test_input = "<script>alert('xss')</script>Hello World"
    sanitized = validator.sanitize_text_input(test_input)
    print(f"Sanitized: {sanitized}")
    
    # Test security report
    report = validator.get_security_report()
    print(f"Security Report: {report}")
