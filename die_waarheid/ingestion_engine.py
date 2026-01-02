"""
Scalable ingestion engine for Die Waarheid
Supports folder-based and ZIP-based import of large datasets (10k–100k files).
"""

import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import hashlib
import logging
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed


class IngestionEngine:
    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.evidence_dir = case_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = case_dir / "manifest.json"
        self.logger = logging.getLogger(__name__)
        
        # Supported file types
        self.audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus'}
        self.text_extensions = {'.txt', '.doc', '.docx', '.pdf', '.rtf'}
        self.chat_keywords = ['whatsapp', 'chat', 'sms', 'messenger', 'telegram']
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension and name."""
        ext = file_path.suffix.lower()
        
        if ext in self.audio_extensions:
            return "audio_statement"
        elif ext in self.text_extensions:
            if any(kw in file_path.name.lower() for kw in self.chat_keywords):
                return "chat_export"
            return "text_statement"
        else:
            return "document"
    
    def ensure_unique_path(self, dest: Path) -> Path:
        """Ensure destination path is unique (avoid overwrites)."""
        if not dest.exists():
            return dest
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        i = 1
        while True:
            candidate = parent / f"{stem} ({i}){suffix}"
            if not candidate.exists():
                return candidate
            i += 1
    
    def copy_file_with_metadata(self, src: Path, relative_path: Optional[str] = None) -> Dict[str, Any]:
        """Copy file to evidence directory and return metadata."""
        try:
            # Determine destination path
            if relative_path:
                dest = self.evidence_dir / relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest = self.evidence_dir / src.name
            
            dest = self.ensure_unique_path(dest)
            
            # Copy file
            shutil.copy2(src, dest)
            
            # Calculate hash
            file_hash = self.calculate_file_hash(dest)
            
            return {
                "original_path": str(src),
                "relative_path": str(dest.relative_to(self.evidence_dir)),
                "full_path": str(dest),
                "name": dest.name,
                "size": dest.stat().st_size,
                "type": self.get_file_type(src),
                "sha256": file_hash,
                "ingested_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to copy {src}: {e}")
            return None
    
    def ingest_folder(self, folder_path: Path, parallel: bool = True) -> List[Dict[str, Any]]:
        """Ingest all supported files from a folder."""
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder: {folder_path}")
        
        # Find all supported files
        files_to_process = []
        for ext in self.audio_extensions.union(self.text_extensions):
            files_to_process.extend(folder_path.rglob(f"*{ext}"))
        
        self.logger.info(f"Found {len(files_to_process)} files to ingest")
        
        if parallel and len(files_to_process) > 100:
            # Parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        self.copy_file_with_metadata,
                        f,
                        f.relative_to(folder_path)
                    )
                    for f in files_to_process
                ]
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            # Sequential processing
            results = []
            for f in files_to_process:
                result = self.copy_file_with_metadata(f, f.relative_to(folder_path))
                if result:
                    results.append(result)
        
        # Update manifest
        self.update_manifest(results)
        
        return results
    
    def ingest_zip(self, zip_path: Path, parallel: bool = True) -> List[Dict[str, Any]]:
        """Ingest files from a ZIP archive."""
        if not zip_path.exists() or not zip_path.suffix.lower() == '.zip':
            raise ValueError(f"Invalid ZIP file: {zip_path}")
        
        temp_dir = self.case_dir / "temp_extract"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Ingest extracted files
            results = self.ingest_folder(temp_dir, parallel=parallel)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process ZIP {zip_path}: {e}")
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def update_manifest(self, new_files: List[Dict[str, Any]]):
        """Update the case manifest with new files."""
        # Load existing manifest
        manifest = {"files": [], "ingested_at": datetime.now().isoformat()}
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
        
        # Add new files (avoid duplicates by SHA-256)
        existing_hashes = {f["sha256"] for f in manifest["files"]}
        for file_info in new_files:
            if file_info["sha256"] not in existing_hashes:
                manifest["files"].append(file_info)
                existing_hashes.add(file_info["sha256"])
        
        # Save updated manifest
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Manifest updated: {len(manifest['files'])} total files")
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get the current case manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {"files": [], "ingested_at": None}
    
    def get_file_list(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of ingested files, optionally filtered by type."""
        manifest = self.get_manifest()
        files = manifest["files"]
        
        if file_type:
            files = [f for f in files if f["type"] == file_type]
        
        return files


def main():
    """CLI interface for ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest files into Die Waarheid case")
    parser.add_argument("--case", type=str, required=True, help="Case name or ID")
    parser.add_argument("--source", type=str, required=True, help="Source folder or ZIP file")
    parser.add_argument("--type", type=str, choices=["folder", "zip"], required=True)
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    
    args = parser.parse_args()
    
    # Setup case directory
    case_dir = Path("data") / "cases" / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ingestion engine
    engine = IngestionEngine(case_dir)
    
    # Ingest files
    source_path = Path(args.source)
    
    if args.type == "folder":
        results = engine.ingest_folder(source_path, parallel=args.parallel)
    elif args.type == "zip":
        results = engine.ingest_zip(source_path, parallel=args.parallel)
    
    print(f"\nIngestion complete!")
    print(f"Files processed: {len(results)}")
    print(f"Case directory: {case_dir}")


if __name__ == "__main__":
    main()
