#!/usr/bin/env python3
"""
Simple Die Waarheid Analysis Runner
Just run: python run_analysis.py
"""

import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 60)
    print("DIE WAARHEID - FORENSIC ANALYSIS SYSTEM")
    print("=" * 60)
    
    try:
        # Import and initialize
        from main_orchestrator import MainOrchestrator
        
        print("\n🚀 Initializing system...")
        orchestrator = MainOrchestrator()
        
        # Create case
        case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        case_name = "Sample Investigation"
        print(f"📁 Creating case: {case_id}")
        orchestrator.create_case(case_id, case_name)
        
        # Add sample evidence (if statement.txt exists)
        sample_file = "statement.txt"
        if os.path.exists(sample_file):
            print(f"📄 Adding evidence: {sample_file}")
            orchestrator.add_evidence("EV_001", "text", sample_file, "Sample statement")
        else:
            print("📄 Creating sample evidence...")
            with open(sample_file, "w") as f:
                f.write("This is a sample statement for analysis.\n")
                f.write("I was at home all night watching TV.\n")
                f.write("The suspect was wearing a black jacket.\n")
                f.write("I saw them leave around 10 PM.\n")
            orchestrator.add_evidence("EV_001", "text", sample_file, "Sample statement")
        
        # Run analysis
        print("\n🔍 Running complete analysis...")
        print("This may take a moment...")
        results = orchestrator.run_complete_analysis()
        
        # Export results
        output_file = f"analysis_results_{case_id}.json"
        print(f"💾 Saving results to: {output_file}")
        orchestrator.export_results(results, output_file)
        
        # Show summary
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if isinstance(results, dict):
            print(f"📊 Results summary:")
            for key, value in results.items():
                if isinstance(value, list):
                    print(f"  • {key}: {len(value)} items")
                else:
                    print(f"  • {key}: {type(value).__name__}")
        
        print(f"\n📁 Full results saved to: {output_file}")
        print(f"📁 Case ID: {case_id}")
        
        # Show key findings if available
        if isinstance(results, dict):
            if 'alerts' in results and results['alerts']:
                print(f"\n🚨 Alerts found: {len(results['alerts'])}")
                for alert in results['alerts'][:3]:  # Show first 3
                    print(f"  • {alert.get('type', 'Unknown')}: {alert.get('message', 'No message')}")
            
            if 'evidence_scoring' in results and results['evidence_scoring']:
                print(f"\n📈 Evidence scored: {len(results['evidence_scoring'])} items")
                for score in results['evidence_scoring'][:3]:  # Show first 3
                    score_val = getattr(score, 'overall_score', 0)
                    print(f"  • Evidence {getattr(score, 'evidence_id', 'Unknown')}: {score_val}/100")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Try installing dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Check the troubleshooting guide in USAGE_GUIDE.md")
        return 1
    
    print("\n🎉 Done! Check your results file.")
    return 0

if __name__ == "__main__":
    exit(main())
