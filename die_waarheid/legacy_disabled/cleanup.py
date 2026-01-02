"""
Disk Cleanup Utility for Die Waarheid
Helps free up disk space by cleaning temporary files, caches, and old analysis results
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

def get_folder_size(folder_path: Path) -> dict:
    """Calculate folder size and file count"""
    total_size = 0
    file_count = 0
    
    if folder_path.exists():
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    pass
    
    return {
        'size_bytes': total_size,
        'size_mb': total_size / (1024 * 1024),
        'size_gb': total_size / (1024 * 1024 * 1024),
        'file_count': file_count
    }

def cleanup_temp_files():
    """Clean temporary files"""
    cleaned = []
    
    # Clean temp directories
    temp_dirs = [
        Path("./temp"),
        Path("./data/temp"),
        Path(os.environ.get('TEMP', '')),
        Path(os.environ.get('TMP', ''))
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            size_before = get_folder_size(temp_dir)
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(exist_ok=True)
                cleaned.append(f"✅ {temp_dir}: {size_before['size_mb']:.1f} MB freed")
            except Exception as e:
                cleaned.append(f"❌ {temp_dir}: Failed to clean ({e})")
    
    # Clean Python cache
    cache_dirs = []
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dirs.append(Path(root) / "__pycache__")
    
    for cache_dir in cache_dirs:
        size_before = get_folder_size(cache_dir)
        try:
            shutil.rmtree(cache_dir)
            cleaned.append(f"✅ Python cache: {size_before['size_mb']:.1f} MB freed")
        except:
            pass
    
    return cleaned

def cleanup_old_exports(days_old: int = 7):
    """Clean export files older than specified days"""
    cleaned = []
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    # Clean old exports
    export_patterns = [
        "./data/output/exports/*.json",
        "./data/output/exports/*.csv",
        "./data/output/exports/*.xlsx",
        "./data/output/reports/*.json",
        "./data/output/reports/*.md",
        "./data/output/mobitables/*.md",
        "./data/*.json",
        "./data/*.csv"
    ]
    
    for pattern in export_patterns:
        for file_path in glob.glob(pattern):
            file_path = Path(file_path)
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned.append(f"✅ {file_path.name}: {size / (1024*1024):.1f} MB")
            except:
                pass
    
    return cleaned

def cleanup_model_cache():
    """Clean model cache files"""
    cleaned = []
    
    # HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        size_before = get_folder_size(hf_cache)
        if size_before['size_gb'] > 5:  # Only clean if > 5GB
            try:
                shutil.rmtree(hf_cache)
                cleaned.append(f"✅ HuggingFace cache: {size_before['size_gb']:.1f} GB freed")
            except Exception as e:
                cleaned.append(f"❌ HuggingFace cache: Failed to clean ({e})")
    
    # Whisper models
    whisper_cache = Path.home() / ".cache" / "whisper"
    if whisper_cache.exists():
        size_before = get_folder_size(whisper_cache)
        try:
            shutil.rmtree(whisper_cache)
            cleaned.append(f"✅ Whisper cache: {size_before['size_mb']:.1f} MB freed")
        except:
            pass
    
    return cleaned

def main():
    """Main cleanup interface"""
    st.set_page_config(page_title="Die Waarheid Cleanup", page_icon="🧹")
    st.markdown("# 🧹 Die Waarheid Disk Cleanup Utility")
    
    # Show current disk usage
    st.markdown("## 📊 Current Disk Usage")
    
    # Check main folders
    folders_to_check = {
        "Data Directory": Path("./data"),
        "Temp Files": Path("./data/temp"),
        "Exports": Path("./data/output/exports"),
        "Reports": Path("./data/output/reports"),
        "Python Cache": Path("./__pycache__") if Path("./__pycache__").exists() else None
    }
    
    total_size = 0
    for name, folder in folders_to_check.items():
        if folder:
            info = get_folder_size(folder)
            total_size += info['size_bytes']
            st.metric(name, f"{info['size_mb']:.1f} MB", f"{info['file_count']} files")
    
    st.metric("Total", f"{total_size / (1024*1024):.1f} MB")
    
    # Cleanup options
    st.markdown("## 🗑️ Cleanup Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clean Temp Files", help="Remove all temporary files"):
            with st.spinner("Cleaning temporary files..."):
                results = cleanup_temp_files()
                for result in results:
                    st.write(result)
                st.success("Temporary files cleaned!")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clean Old Exports", help="Remove exports older than 7 days"):
            with st.spinner("Cleaning old exports..."):
                results = cleanup_old_exports()
                for result in results:
                    st.write(result)
                st.success("Old exports cleaned!")
                st.rerun()
    
    with col3:
        if st.button("🤖 Clean Model Cache", help="Remove downloaded AI models"):
            with st.spinner("Cleaning model cache..."):
                results = cleanup_model_cache()
                for result in results:
                    st.write(result)
                st.success("Model cache cleaned!")
                st.rerun()
    
    # Full cleanup
    st.markdown("---")
    if st.button("🚨 FULL CLEANUP (All of the above)", type="secondary"):
        if st.checkbox("I understand this will delete all temporary files, old exports, and model caches"):
            with st.spinner("Performing full cleanup..."):
                all_results = []
                all_results.extend(cleanup_temp_files())
                all_results.extend(cleanup_old_exports())
                all_results.extend(cleanup_model_cache())
                
                st.write("## Cleanup Results:")
                for result in all_results:
                    st.write(result)
                
                st.success("Full cleanup complete!")
                st.rerun()
    
    # Manual cleanup
    st.markdown("## 📁 Manual Cleanup")
    
    folder_to_clean = st.selectbox(
        "Select folder to clean completely:",
        ["./data/temp", "./data/output/exports", "./data/output/reports", "./data/logs"]
    )
    
    if st.button(f"Delete all files in {folder_to_clean}"):
        folder_path = Path(folder_to_clean)
        if folder_path.exists():
            size_before = get_folder_size(folder_path)
            try:
                shutil.rmtree(folder_path)
                folder_path.mkdir(exist_ok=True)
                st.success(f"Deleted {size_before['size_mb']:.1f} MB from {folder_to_clean}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clean {folder_to_clean}: {e}")

if __name__ == "__main__":
    main()
