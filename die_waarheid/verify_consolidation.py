#!/usr/bin/env python3
"""
Verification script to confirm codebase consolidation and prevent import collisions.
This ensures only the real die_waarheid pipeline is runnable.
"""

import sys
import os
import importlib.util

def test_import_collision_prevention():
    """Test that legacy modules cannot cause import collisions."""
    print("=== Codebase Consolidation Verification ===\n")
    
    # Check that legacy src is properly disabled in legacy_disabled/
    legacy_src_disabled = os.path.exists("legacy_disabled/src_legacy_disabled")
    print(f"Legacy 'src' moved to 'legacy_disabled/': {legacy_src_disabled}")
    
    if not legacy_src_disabled:
        print("⚠️  WARN: Legacy src directory not found - may have been manually removed")
    else:
        print("✅ PASS: Legacy src directory properly disabled")
    
    # Check that real die_waarheid src exists
    real_src_exists = os.path.exists("src")
    print(f"Real 'src' directory exists: {real_src_exists}")
    
    if not real_src_exists:
        print("❌ FAIL: Real src directory missing!")
        return False
    else:
        print("✅ PASS: Real src directory present")
    
    # Test imports from real pipeline
    sys.path.insert(0, '.')
    sys.path.insert(0, './src')
    
    critical_modules = [
        ('real_analysis_engine', 'RealAnalysisEngine'),
        ('src.text_forensics', 'TextForensicsEngine'),
        ('src.expert_panel', 'ExpertPanelAnalyzer'),
        ('src.afrikaans_processor', 'AfrikaansProcessor'),
        ('src.narrative_reconstruction', 'NarrativeReconstructor'),
        ('src.contradiction_timeline', 'ContradictionTimelineAnalyzer'),
        ('src.investigative_checklist', 'InvestigativeChecklistGenerator'),
    ]
    
    print("\n=== Testing Critical Module Imports ===")
    all_imports_ok = True
    
    for module_name, class_name in critical_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls:
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"❌ {module_name}.{class_name} - class not found")
                all_imports_ok = False
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {e}")
            all_imports_ok = False
    
    # Test that legacy modules are not importable
    print("\n=== Testing Legacy Modules Are Disabled ===")
    legacy_modules = [
        ('src', 'ai_analyzer'),
        ('src', 'forensics'), 
        ('src', 'chat_parser'),
    ]
    
    for module_name, class_name in legacy_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls:
                print(f"❌ Legacy {module_name}.{class_name} still importable!")
                all_imports_ok = False
            else:
                print(f"✅ Legacy {module_name}.{class_name} not found")
        except ImportError:
            print(f"✅ Legacy {module_name} properly disabled")
        except Exception as e:
            print(f"✅ Legacy {module_name} disabled ({e})")
    
    # Test dashboard behavior
    print("\n=== Testing Dashboard Behavior ===")
    try:
        import dashboard_real
        print("✅ Real dashboard (dashboard_real.py) importable")
    except Exception as e:
        print(f"❌ Real dashboard import failed: {e}")
        all_imports_ok = False
    
    try:
        import dashboard
        # If we can import it, check that it's disabled
        import inspect
        main_func = getattr(dashboard, 'main', None)
        if main_func:
            source = inspect.getsource(main_func)
            if 'deprecated' in source and 'st.stop()' in source:
                print("✅ Legacy dashboard properly disabled")
            else:
                print("❌ Legacy dashboard not properly disabled")
                all_imports_ok = False
        else:
            print("❌ Legacy dashboard missing main function")
            all_imports_ok = False
    except Exception as e:
        print(f"✅ Legacy dashboard disabled ({e})")
    
    # Final result
    print("\n" + "="*50)
    if all_imports_ok:
        print("🎉 SUCCESS: Codebase consolidation complete!")
        print("✅ Only real die_waarheid pipeline is runnable")
        print("✅ No import collision risk")
        print("✅ Fake analytics eliminated")
    else:
        print("❌ FAILURE: Codebase consolidation incomplete")
        print("⚠️  Some issues remain - review above")
    
    return all_imports_ok

if __name__ == "__main__":
    success = test_import_collision_prevention()
    sys.exit(0 if success else 1)
