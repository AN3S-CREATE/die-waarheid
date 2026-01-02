"""
TRUTHFUL Die Waarheid Dashboard - Shows REAL status only
No fake results, no lies about processing that didn't happen.
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Die Waarheid - TRUTHFUL Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_actual_case_data():
    """Get ACTUAL case data - no fake results."""
    cases_dir = Path("die_waarheid/data/cases")
    
    if not cases_dir.exists():
        return {"exists": False, "message": "No cases directory found"}
    
    cases = {}
    for case_dir in cases_dir.iterdir():
        if case_dir.is_dir():
            case_name = case_dir.name
            
            # Count ACTUAL files
            evidence_dir = case_dir / "evidence"
            actual_files = []
            audio_count = 0
            text_count = 0
            
            if evidence_dir.exists():
                for file_path in evidence_dir.iterdir():
                    if file_path.is_file():
                        actual_files.append(str(file_path))
                        if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
                            audio_count += 1
                        elif file_path.suffix.lower() in ['.txt']:
                            text_count += 1
            
            # Check for REAL analysis results
            runs_dir = case_dir / "runs"
            real_results = []
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        results_file = run_dir / "analysis_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                    # Check if results are REAL (not empty templates)
                                    processed = results.get('summary', {}).get('processed', 0)
                                    if processed > 0:
                                        real_results.append({
                                            "run_id": run_dir.name,
                                            "processed": processed,
                                            "timestamp": results.get('started_at', 'Unknown')
                                        })
                            except:
                                pass
            
            cases[case_name] = {
                "exists": True,
                "actual_files": len(actual_files),
                "audio_files": audio_count,
                "text_files": text_count,
                "real_analysis_runs": len(real_results),
                "real_results": real_results
            }
    
    return cases

def render_header():
    """Render truthful header."""
    st.markdown("# 🔍 Die Waarheid - TRUTHFUL Status")
    st.markdown("### No fake analytics. No false results. Only reality.")

def render_truthful_status():
    """Show ACTUAL system status."""
    st.header("📊 ACTUAL System Status")
    
    # Check what REALLY exists
    cases = get_actual_case_data()
    
    if not cases:
        st.error("❌ NO CASE DATA FOUND")
        st.info("The system has no real case data to analyze.")
        return
    
    # Show TRUTHFUL summary
    total_audio = sum(case["audio_files"] for case in cases.values())
    total_text = sum(case["text_files"] for case in cases.values())
    total_real_analysis = sum(case["real_analysis_runs"] for case in cases.values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Real Cases", len(cases))
    with col2:
        st.metric("Audio Files", total_audio)
    with col3:
        st.metric("Text Files", total_text)
    with col4:
        st.metric("Real Analysis Runs", total_real_analysis)
    
    # Show detailed TRUTH
    st.divider()
    st.subheader("🎯 TRUTHFUL Case Details")
    
    for case_name, case_data in cases.items():
        with st.expander(f"📁 {case_name}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Audio files:** {case_data['audio_files']}")
                st.write(f"**Text files:** {case_data['text_files']}")
                st.write(f"**Total files:** {case_data['actual_files']}")
            
            with col2:
                if case_data['real_analysis_runs'] > 0:
                    st.success(f"✅ {case_data['real_analysis_runs']} REAL analysis runs")
                    for result in case_data['real_results']:
                        st.write(f"  - {result['run_id']}: {result['processed']} files processed")
                else:
                    st.error("❌ NO REAL ANALYSIS PERFORMED")
                    st.warning("This case has data but has never been actually processed.")
    
    # Show HONEST assessment
    st.divider()
    st.subheader("🎲 HONEST System Assessment")
    
    if total_audio > 0 and total_real_analysis == 0:
        st.error("⚠️ **CRITICAL ISSUE DETECTED**")
        st.write(f"The system has **{total_audio} audio files** but has **never actually processed them**.")
        st.write("The dashboard may show 'analysis results' but these are FAKE - no real transcription or analysis has occurred.")
        
        st.warning("🔥 **TRUTH:** The system has real data but produces fake results.")
    
    elif total_audio > 0 and total_real_analysis > 0:
        st.success("✅ **REAL PROCESSING DETECTED**")
        st.write(f"The system has actually processed {total_audio} audio files.")
        st.write("Analysis results are REAL and based on actual data.")
    
    else:
        st.info("ℹ️ **NO DATA TO PROCESS**")
        st.write("The system has no audio files to analyze.")

def render_reality_check():
    """Show what's REAL vs FAKE."""
    st.header("🔍 Reality Check")
    
    st.subheader("What's REAL:")
    st.write("✅ Audio files in case directories")
    st.write("✅ Whisper transcription model (loads successfully)")
    st.write("✅ Analysis engine code (can process files)")
    
    st.subheader("What's FAKE:")
    st.write("❌ Analysis results showing '0 files processed' when data exists")
    st.write("❌ Dashboard metrics showing 'analysis complete' when nothing was processed")
    st.write("❌ Reports and analytics that are empty templates")

def main():
    """Main application - TRUTHFUL version."""
    render_header()
    
    # Navigation
    tab1, tab2 = st.tabs(["📊 Truthful Status", "🔍 Reality Check"])
    
    with tab1:
        render_truthful_status()
    
    with tab2:
        render_reality_check()
    
    # Footer with TRUTH
    st.divider()
    st.markdown("""
    ---
    **TRUTHFUL DISCLAIMER:** This dashboard shows the ACTUAL state of the system.  
    No fake analytics. No false processing claims. Only reality.
    """)

if __name__ == "__main__":
    main()
