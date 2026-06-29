#!/usr/bin/env python3
"""
Validation Script for Status Code 410 Fix
Checks if all changes have been properly applied
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_file_content(filepath, search_string, description):
    """Check if file contains expected content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - Expected content not found")
                return False
    except Exception as e:
        print(f"❌ {description} - Error reading file: {e}")
        return False

def main():
    print("=" * 70)
    print("Status Code 410 Fix Validation")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check 1: Configuration file updates
    print("📋 Checking Configuration Files...")
    print("-" * 70)
    
    config_checks = [
        ("die_waarheid/config.py", 'GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")', 
         "Config uses gemini-1.5-flash model"),
        ("requirements.txt", "google-generativeai==0.8.3", 
         "Requirements has updated google-generativeai"),
        (".env.example", "GEMINI_MODEL=gemini-1.5-flash", 
         ".env.example has GEMINI_MODEL setting"),
        (".env.example", "USE_FREE_AI=true", 
         ".env.example has USE_FREE_AI setting"),
    ]
    
    for filepath, search_str, desc in config_checks:
        if not check_file_content(filepath, search_str, desc):
            all_checks_passed = False
    
    print()
    
    # Check 2: Error handling updates
    print("🛡️ Checking Error Handling Updates...")
    print("-" * 70)
    
    error_handling_checks = [
        ("die_waarheid/src/ai_analyzer.py", "'410' in error_str", 
         "ai_analyzer.py has 410 error detection"),
        ("die_waarheid/src/text_forensics.py", "'410' in error_str", 
         "text_forensics.py has 410 error detection"),
        ("die_waarheid/src/expert_panel.py", "'410' in error_str", 
         "expert_panel.py has 410 error detection"),
        ("die_waarheid/src/afrikaans_processor.py", "'410' in error_str", 
         "afrikaans_processor.py has 410 error detection"),
        ("die_waarheid/src/afrikaans_fallback.py", "'410' in error_str", 
         "afrikaans_fallback.py has 410 error detection"),
    ]
    
    for filepath, search_str, desc in error_handling_checks:
        if not check_file_content(filepath, search_str, desc):
            all_checks_passed = False
    
    print()
    
    # Check 3: Documentation files
    print("📚 Checking Documentation Files...")
    print("-" * 70)
    
    doc_checks = [
        ("BLACKBOX_CLI_FIX.md", "Documentation for technical fix"),
        ("QUICK_START.md", "Quick start guide"),
        ("STATUS_410_FIX_SUMMARY.md", "Complete fix summary"),
    ]
    
    for filepath, desc in doc_checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    print()
    
    # Check 4: Key error handling patterns
    print("🔍 Checking Error Handling Patterns...")
    print("-" * 70)
    
    pattern_checks = [
        ("die_waarheid/src/ai_analyzer.py", "error_type", 
         "ai_analyzer.py returns error_type"),
        ("die_waarheid/src/ai_analyzer.py", "deprecated_model", 
         "ai_analyzer.py handles deprecated_model error"),
        ("die_waarheid/src/text_forensics.py", "deprecated", 
         "text_forensics.py checks for deprecated"),
    ]
    
    for filepath, search_str, desc in pattern_checks:
        if not check_file_content(filepath, search_str, desc):
            all_checks_passed = False
    
    print()
    
    # Check 5: Import test (optional)
    print("🐍 Checking Python Imports...")
    print("-" * 70)
    
    try:
        # Try to import config to verify syntax
        sys.path.insert(0, str(Path(__file__).parent))
        from die_waarheid import config
        print(f"✅ Config imports successfully")
        print(f"   - GEMINI_MODEL: {config.GEMINI_MODEL}")
        print(f"   - USE_FREE_AI: {config.USE_FREE_AI}")
    except ImportError as e:
        print(f"⚠️ Config import test skipped (dependencies not installed)")
        print(f"   Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        all_checks_passed = False
    
    print()
    
    # Final summary
    print("=" * 70)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("The Status Code 410 fix has been successfully applied.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure .env file (copy from .env.example)")
        print("3. Run the application: python3 die_waarheid/launcher.py")
        print()
        print("For more information, see:")
        print("  - QUICK_START.md - Installation and usage guide")
        print("  - BLACKBOX_CLI_FIX.md - Technical details")
        print("  - STATUS_410_FIX_SUMMARY.md - Complete summary")
    else:
        print("⚠️ SOME CHECKS FAILED")
        print()
        print("Please review the failed checks above and ensure all")
        print("changes have been properly applied.")
        print()
        print("For assistance, see:")
        print("  - STATUS_410_FIX_SUMMARY.md - List of all changes")
        print("  - BLACKBOX_CLI_FIX.md - Detailed fix information")
    print("=" * 70)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
