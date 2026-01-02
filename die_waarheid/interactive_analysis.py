#!/usr/bin/env python3
"""
Interactive Die Waarheid Analysis
Actually asks for files and shows what's happening
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_header():
    print("=" * 80)
    print("DIE WAARHEID - INTERACTIVE FORENSIC ANALYSIS")
    print("=" * 80)
    print("This will analyze your actual files and show you the results")
    print()

def get_user_files():
    """Get files from user"""
    files = []
    
    print("📁 What files do you want to analyze?")
    print("   (Enter file paths one by one, press Enter with empty line to finish)")
    print()
    
    while True:
        file_path = input(f"File #{len(files)+1} (or press Enter to finish): ").strip()
        
        if not file_path:
            if len(files) == 0:
                print("⚠️  No files entered! Let's try again...")
                continue
            break
        
        # Check if file exists
        if os.path.exists(file_path):
            files.append(file_path)
            print(f"   ✅ Added: {file_path}")
        else:
            print(f"   ❌ File not found: {file_path}")
            print("   Try again or enter a different file path")
    
    return files

def analyze_files(files):
    """Analyze the provided files"""
    try:
        from main_orchestrator import MainOrchestrator
        
        print("\n🚀 Initializing Die Waarheid...")
        orchestrator = MainOrchestrator()
        
        # Create case
        case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        case_name = f"Analysis of {len(files)} files"
        print(f"📁 Creating case: {case_id}")
        orchestrator.create_case(case_id, case_name)
        
        # Add each file
        print(f"\n📄 Adding {len(files)} files to analysis:")
        for i, file_path in enumerate(files, 1):
            file_name = os.path.basename(file_path)
            
            # Determine file type
            if file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                file_type = "audio"
            elif file_path.lower().endswith(('.txt', '.doc', '.docx')):
                file_type = "text"
            elif 'chat' in file_name.lower() or 'whatsapp' in file_name.lower():
                file_type = "chat_export"
            else:
                file_type = "text"  # default
            
            print(f"   {i}. {file_name} ({file_type})")
            evidence_id = f"EV_{i:03d}"
            orchestrator.add_evidence(evidence_id, file_type, file_path, file_name)
        
        # Run analysis with progress
        print(f"\n🔍 Running complete forensic analysis...")
        print("   This will run all 12 analysis stages...")
        print("   Please wait...")
        
        results = orchestrator.run_complete_analysis()
        
        return results, case_id, orchestrator
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None, None, None

def show_results_summary(results):
    """Show a summary of the analysis results"""
    if not results or not isinstance(results, dict):
        print("❌ No results to display")
        return
    
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    
    # Show case info
    print(f"📁 Case ID: {results.get('case_id', 'Unknown')}")
    print(f"📝 Case Name: {results.get('case_name', 'Unknown')}")
    print(f"⏰ Started: {results.get('started_at', 'Unknown')}")
    print(f"✅ Completed: {results.get('completed_at', 'Unknown')}")
    
    # Show stages completed
    stages = results.get('stages', {})
    print(f"\n🔍 Analysis Stages Completed: {len(stages)}/12")
    
    for stage_name, stage_result in stages.items():
        status = "✅" if stage_result else "❌"
        stage_display = stage_name.replace('_', ' ').title()
        print(f"   {status} {stage_display}")
    
    # Show key findings
    print(f"\n🎯 KEY FINDINGS:")
    
    # Check for alerts
    if 'alerts' in stages and stages['alerts']:
        alerts = stages['alerts']
        if isinstance(alerts, list) and len(alerts) > 0:
            print(f"   🚨 Alerts Found: {len(alerts)}")
            for alert in alerts[:3]:  # Show first 3
                if hasattr(alert, 'type'):
                    print(f"      • {alert.type}: {getattr(alert, 'message', 'No message')}")
                elif isinstance(alert, dict):
                    print(f"      • {alert.get('type', 'Unknown')}: {alert.get('message', 'No message')}")
    
    # Check for evidence scoring
    if 'evidence_scoring' in stages and stages['evidence_scoring']:
        scores = stages['evidence_scoring']
        if isinstance(scores, list) and len(scores) > 0:
            print(f"   📈 Evidence Scored: {len(scores)} items")
            for score in scores[:3]:  # Show first 3
                if hasattr(score, 'overall_score'):
                    print(f"      • {getattr(score, 'evidence_id', 'Unknown')}: {score.overall_score}/100")
                elif isinstance(score, dict):
                    print(f"      • {score.get('evidence_id', 'Unknown')}: {score.get('overall_score', 0)}/100")
    
    # Check for contradictions
    if 'contradiction_timeline' in stages and stages['contradiction_timeline']:
        contradictions = stages['contradiction_timeline']
        if hasattr(contradictions, 'contradictions') and contradictions.contradictions:
            print(f"   ⚠️  Contradictions Found: {len(contradictions.contradictions)}")
        elif isinstance(contradictions, dict) and contradictions.get('contradictions'):
            print(f"   ⚠️  Contradictions Found: {len(contradictions['contradictions'])}")
    
    # Check for investigative checklist
    if 'investigative_checklist' in stages and stages['investigative_checklist']:
        checklist = stages['investigative_checklist']
        if isinstance(checklist, list) and len(checklist) > 0:
            print(f"   📋 Investigative Actions: {len(checklist)} items")
            for item in checklist[:3]:  # Show first 3
                if hasattr(item, 'action'):
                    print(f"      • {item.action}")
                elif isinstance(item, dict):
                    print(f"      • {item.get('action', 'Unknown action')}")

def save_and_show_results(results, case_id, orchestrator):
    """Save results and show file location"""
    if not results:
        return
    
    # Save results
    output_file = f"analysis_results_{case_id}.json"
    print(f"\n💾 Saving results to: {output_file}")
    
    try:
        orchestrator.export_results(results, output_file)
        print(f"   ✅ Results saved successfully!")
        
        # Show file location
        full_path = os.path.abspath(output_file)
        print(f"   📁 Full path: {full_path}")
        
        # Ask if user wants to open the file
        print(f"\n📖 Would you like to see the detailed results?")
        choice = input("   Open results file? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            try:
                # Try to open with default application
                if os.name == 'nt':  # Windows
                    os.startfile(full_path)
                elif os.name == 'posix':  # macOS/Linux
                    os.system(f'open "{full_path}"' if sys.platform == 'darwin' else f'xdg-open "{full_path}"')
                print(f"   📂 Opening results file...")
            except:
                print(f"   ⚠️  Could not open file automatically")
                print(f"   📁 Please open manually: {full_path}")
        
        # Show key findings from JSON
        print(f"\n🔍 QUICK LOOK AT RESULTS:")
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            print(f"   📊 Case: {data.get('case_name', 'Unknown')}")
            print(f"   ⏰ Analysis time: {data.get('started_at', 'Unknown')} to {data.get('completed_at', 'Unknown')}")
            
            stages = data.get('stages', {})
            print(f"   🔍 Stages completed: {len(stages)}/12")
            
            # Show some actual data
            for stage_name, stage_data in stages.items():
                if stage_data and isinstance(stage_data, list) and len(stage_data) > 0:
                    print(f"   📋 {stage_name.replace('_', ' ').title()}: {len(stage_data)} items found")
                    
        except Exception as e:
            print(f"   ⚠️  Could not read results file: {e}")
            
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")

def main():
    """Main interactive function"""
    print_header()
    
    # Get files from user
    files = get_user_files()
    
    if not files:
        print("\n❌ No files provided. Exiting...")
        return 1
    
    print(f"\n🎯 Ready to analyze {len(files)} files:")
    for i, file_path in enumerate(files, 1):
        print(f"   {i}. {file_path}")
    
    # Confirm
    print(f"\n🚀 Starting analysis...")
    
    # Analyze files
    results, case_id, orchestrator = analyze_files(files)
    
    if not results:
        print("\n❌ Analysis failed. Check the error messages above.")
        return 1
    
    # Show summary
    show_results_summary(results)
    
    # Save and show results
    save_and_show_results(results, case_id, orchestrator)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Your results are saved in the analysis results file")
    print(f"🔍 Check the JSON file for detailed findings")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Analysis cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
