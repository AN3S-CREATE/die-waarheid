#!/usr/bin/env python3
"""
Comprehensive test script to validate all fixes in Die Waarheid application
Tests imports, configuration, and core functionality
"""

import sys
import traceback
from pathlib import Path
from typing import List, Tuple

# Add current directory to path
sys.path.insert(0, '.')

def test_critical_imports() -> Tuple[bool, List[str]]:
    """Test all critical module imports"""
    print("🔍 Testing critical imports...")
    
    imports_to_test = [
        ("config", "validate_config, get_config_summary"),
        ("src.logging_config", "JSONFormatter"),
        ("src.models", "ForensicsResult, Message"),
        ("src.gdrive_handler", "GDriveHandler"),
        ("src.chat_parser", "WhatsAppParser"),
        ("src.whisper_transcriber", "WhisperTranscriber"),
        ("src.forensics", "ForensicsEngine"),
        ("src.ai_analyzer", "AIAnalyzer"),
        ("src.background_audio", "analyze_background_audio"),
        ("app", "main"),
    ]
    
    results = []
    all_passed = True
    
    for module, items in imports_to_test:
        try:
            exec(f"from {module} import {items}")
            results.append(f"✅ {module}: {items}")
        except Exception as e:
            results.append(f"❌ {module}: {items} - {str(e)}")
            all_passed = False
    
    return all_passed, results

def test_pydantic_models() -> Tuple[bool, List[str]]:
    """Test Pydantic models for validation errors"""
    print("🔍 Testing Pydantic models...")
    
    results = []
    all_passed = True
    
    try:
        from src.models import ForensicsResult, Message, PsychologicalProfile
        
        # Test ForensicsResult creation with all required fields
        from src.models import IntensityMetrics
        intensity = IntensityMetrics(mean=-20.0, max=-10.0, std=5.0)
        result = ForensicsResult(
            success=True,
            filename="test.txt",
            duration=10.0,
            pitch_volatility=25.0,
            silence_ratio=0.1,
            intensity=intensity,
            mfcc_variance=0.5,
            zero_crossing_rate=0.1,
            spectral_centroid=1000.0,
            stress_level=30.0,
            stress_threshold_exceeded=False,
            high_cognitive_load=False,
            analysis_type="audio",
            confidence=0.85
        )
        results.append("✅ ForensicsResult model works")
        
        # Test Message creation
        message = Message(
            sender="TestUser",
            timestamp="2024-01-01 12:00:00",
            text="Test message",
            message_type="text"
        )
        results.append("✅ Message model works")
        
        # Test PsychologicalProfile creation
        profile = PsychologicalProfile(
            success=True,
            emotional_regulation="moderate",
            risk_assessment="low"
        )
        results.append("✅ PsychologicalProfile model works")
        
    except Exception as e:
        results.append(f"❌ Pydantic models failed: {str(e)}")
        all_passed = False
    
    return all_passed, results

def test_configuration() -> Tuple[bool, List[str]]:
    """Test configuration validation"""
    print("🔍 Testing configuration...")
    
    results = []
    all_passed = True
    
    try:
        from config import validate_config, get_config_summary
        
        errors, warnings = validate_config()
        
        results.append(f"📊 Configuration errors: {len(errors)}")
        for error in errors:
            results.append(f"  ❌ {error}")
        
        results.append(f"📊 Configuration warnings: {len(warnings)}")
        for warning in warnings:
            results.append(f"  ⚠️ {warning}")
        
        # Configuration is valid if we can load it without exceptions
        results.append("✅ Configuration module loads successfully")
        
    except Exception as e:
        results.append(f"❌ Configuration test failed: {str(e)}")
        all_passed = False
    
    return all_passed, results

def test_file_structure() -> Tuple[bool, List[str]]:
    """Test required file and directory structure"""
    print("🔍 Testing file structure...")
    
    results = []
    all_passed = True
    
    required_files = [
        ".env",
        ".env.example", 
        "config.py",
        "app.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "src",
        "data",
        "data/audio",
        "data/text", 
        "data/temp",
        "data/output",
        "credentials"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            results.append(f"✅ File exists: {file_path}")
        else:
            results.append(f"❌ Missing file: {file_path}")
            all_passed = False
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            results.append(f"✅ Directory exists: {dir_path}")
        else:
            results.append(f"❌ Missing directory: {dir_path}")
            all_passed = False
    
    return all_passed, results

def test_main_application() -> Tuple[bool, List[str]]:
    """Test main application can be imported and has required functions"""
    print("🔍 Testing main application...")
    
    results = []
    all_passed = True
    
    try:
        import app
        
        if hasattr(app, 'main'):
            results.append("✅ main() function found")
        else:
            results.append("❌ main() function not found")
            all_passed = False
        
        # Check for key functions
        key_functions = ['render_sidebar', 'render_upload_page', 'render_analysis_page']
        for func_name in key_functions:
            if hasattr(app, func_name):
                results.append(f"✅ {func_name}() function found")
            else:
                results.append(f"⚠️ {func_name}() function not found")
        
    except Exception as e:
        results.append(f"❌ Main application test failed: {str(e)}")
        all_passed = False
    
    return all_passed, results

def test_deprecated_warnings() -> Tuple[bool, List[str]]:
    """Test that deprecated warnings are properly handled"""
    print("🔍 Testing deprecated warning handling...")
    
    results = []
    all_passed = True
    
    try:
        import warnings
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import modules that might generate deprecation warnings
            from src.ai_analyzer import AIAnalyzer
            
            # Check if deprecation warnings are filtered
            genai_warnings = [warning for warning in w if 'google.generativeai' in str(warning.message)]
            
            if genai_warnings:
                results.append(f"⚠️ {len(genai_warnings)} deprecation warnings still showing")
            else:
                results.append("✅ Deprecation warnings properly filtered")
        
    except Exception as e:
        results.append(f"❌ Deprecation warning test failed: {str(e)}")
        all_passed = False
    
    return all_passed, results

def main():
    """Run all tests and report results"""
    print("🚀 Die Waarheid - Comprehensive Fix Validation")
    print("=" * 60)
    
    tests = [
        ("Critical Imports", test_critical_imports),
        ("Pydantic Models", test_pydantic_models),
        ("Configuration", test_configuration),
        ("File Structure", test_file_structure),
        ("Main Application", test_main_application),
        ("Deprecated Warnings", test_deprecated_warnings),
    ]
    
    all_tests_passed = True
    total_issues = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            passed, results = test_func()
            
            for result in results:
                print(result)
                if result.startswith("❌"):
                    total_issues += 1
            
            if not passed:
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            print(traceback.format_exc())
            all_tests_passed = False
            total_issues += 1
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if all_tests_passed and total_issues == 0:
        print("🎉 ALL TESTS PASSED! Die Waarheid is ready to use.")
    elif total_issues <= 3:
        print(f"⚠️ Minor issues found ({total_issues}). Application should work with limited functionality.")
    else:
        print(f"❌ Significant issues found ({total_issues}). Manual intervention required.")
    
    print(f"\nTotal issues found: {total_issues}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
