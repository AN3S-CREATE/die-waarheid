"""
Verification harness for Die Waarheid forensic pipeline
Produces audit-grade outputs with provenance, hashes, and reproducibility.
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import traceback

# Import real pipeline modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.whisper_transcriber import WhisperTranscriber
from src.afrikaans_processor import AfrikaansProcessor
from src.afrikaans_audio import AudioLayerSeparator
from src.diarization import SimpleDiarizer
from src.multilingual_support import MultilingualAnalyzer


class VerificationHarness:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize real modules
        self.whisper = WhisperTranscriber()
        self.afrikaans_processor = AfrikaansProcessor()
        self.audio_separator = AudioLayerSeparator()
        self.diarizer = SimpleDiarizer()
        self.multilingual = MultilingualAnalyzer()
    
    def setup_logging(self):
        log_file = self.run_dir / "process.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def load_file_list(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load and hash all input files."""
        inputs = []
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                continue
            
            file_info = {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
                "sha256": self.calculate_file_hash(path)
            }
            inputs.append(file_info)
        
        # Save input manifest
        with open(self.run_dir / "inputs.json", "w", encoding="utf-8") as f:
            json.dump(inputs, f, indent=2, ensure_ascii=False)
        
        return inputs
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration snapshot."""
        config["timestamp"] = datetime.now().isoformat()
        config["python_version"] = sys.version
        config["modules"] = {
            "whisper_transcriber": "Whisper-based transcription with Afrikaans optimization",
            "afrikaans_processor": "Triple-check verification for Afrikaans text",
            "audio_layer_separator": "Foreground/background audio separation",
            "diarization": "Energy-based speaker segmentation",
            "multilingual_support": "Code-switch detection (Afrikaans/English)"
        }
        
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def process_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file through the full pipeline."""
        file_path = Path(file_info["path"])
        result = {
            "file": file_info["name"],
            "sha256": file_info["sha256"],
            "status": "processing",
            "stages": {},
            "errors": []
        }
        
        try:
            # Stage 1: Transcription (Whisper + Afrikaans optimization)
            self.logger.info(f"Transcribing {file_path.name}...")
            raw_transcript = self.whisper.transcribe(str(file_path))
            result["stages"]["whisper_transcription"] = {
                "status": "success",
                "text": raw_transcript.get("text", ""),
                "language": raw_transcript.get("language", "unknown"),
                "confidence": raw_transcript.get("confidence", 0)
            }
            
            # Stage 2: Afrikaans processing (triple-check verification)
            self.logger.info(f"Processing Afrikaans for {file_path.name}...")
            afrikaans_result = self.afrikaans_processor.process_text(raw_transcript.get("text", ""))
            result["stages"]["afrikaans_processing"] = {
                "status": "success",
                "verified_text": afrikaans_result.get("verified_text", ""),
                "confidence": afrikaans_result.get("confidence", 0),
                "corrections": afrikaans_result.get("corrections", [])
            }
            
            # Stage 3: Audio layer separation (foreground/background)
            self.logger.info(f"Separating audio layers for {file_path.name}...")
            layers = self.audio_separator.separate_layers(str(file_path))
            result["stages"]["audio_separation"] = {
                "status": "success",
                "foreground_detected": layers.get("foreground_detected", False),
                "background_noise": layers.get("background_noise", {}),
                "voice_activity": layers.get("voice_activity", [])
            }
            
            # Stage 4: Diarization (speaker segmentation)
            self.logger.info(f"Running diarization for {file_path.name}...")
            diarization = self.diarizer.diarize(str(file_path))
            result["stages"]["diarization"] = {
                "status": "success",
                "segments": diarization.get("segments", []),
                "speaker_count": diarization.get("speaker_count", 0)
            }
            
            # Stage 5: Multilingual analysis (code-switch detection)
            self.logger.info(f"Analyzing language patterns for {file_path.name}...")
            multilingual = self.multilingual.analyze(raw_transcript.get("text", ""))
            result["stages"]["multilingual_analysis"] = {
                "status": "success",
                "primary_language": multilingual.get("primary_language", "unknown"),
                "language_switches": multilingual.get("language_switches", []),
                "code_switches": multilingual.get("code_switches", [])
            }
            
            # Background sound identification
            background_sounds = self.identify_background_sounds(layers.get("background_noise", {}))
            result["stages"]["background_sounds"] = {
                "status": "success",
                "identified_sounds": background_sounds
            }
            
            result["status"] = "completed"
            result["final_transcript"] = afrikaans_result.get("verified_text", raw_transcript.get("text", ""))
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            self.logger.error(f"Failed to process {file_path.name}: {e}")
            self.logger.error(traceback.format_exc())
        
        return result
    
    def identify_background_sounds(self, background_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify and classify background sounds."""
        sounds = []
        
        # Simple heuristics for sound identification
        if background_data.get("energy_level", 0) > 0.7:
            sounds.append({
                "type": "loud_noise",
                "confidence": 0.8,
                "description": "High energy background noise detected"
            })
        
        if background_data.get("frequency_profile", {}).get("low_freq_dominant", False):
            sounds.append({
                "type": "low_frequency_noise",
                "confidence": 0.7,
                "description": "Low frequency noise (possibly traffic, machinery)"
            })
        
        # Add more sophisticated sound identification here
        # This could integrate with audio classification models
        
        return sounds
    
    def run_verification(self, file_paths: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run full verification on selected files."""
        self.logger.info(f"Starting verification run on {len(file_paths)} files")
        
        # Save configuration
        default_config = {
            "primary_language": "afrikaans",
            "secondary_languages": ["english"],
            "enable_background_analysis": True,
            "enable_code_switch_detection": True,
            "verification_mode": "forensic"
        }
        if config:
            default_config.update(config)
        self.save_config(default_config)
        
        # Load and hash input files
        inputs = self.load_file_list(file_paths)
        
        # Process files
        results = []
        start_time = time.time()
        
        for i, file_info in enumerate(inputs, 1):
            self.logger.info(f"Processing file {i}/{len(inputs)}: {file_info['name']}")
            result = self.process_file(file_info)
            results.append(result)
            
            # Save intermediate results
            with open(self.run_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        end_time = time.time()
        summary = {
            "run_id": self.run_dir.name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": end_time - start_time,
            "total_files": len(inputs),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "primary_language": default_config["primary_language"],
            "background_analysis_enabled": default_config["enable_background_analysis"],
            "files": results
        }
        
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Verification complete. {summary['completed']}/{summary['total_files']} files processed successfully.")
        
        return summary


def main():
    """CLI interface for verification harness."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Die Waarheid verification harness")
    parser.add_argument("--files", nargs="+", required=True, help="List of files to process")
    parser.add_argument("--output", type=str, default="data/verification", help="Output directory")
    parser.add_argument("--config", type=str, help="Config file (JSON)")
    
    args = parser.parse_args()
    
    # Create run directory with timestamp
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output) / run_id
    
    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = json.load(f)
    
    # Run verification
    harness = VerificationHarness(run_dir)
    results = harness.run_verification(args.files, config)
    
    print(f"\nVerification complete!")
    print(f"Results saved to: {run_dir}")
    print(f"Completed: {results['completed']}/{results['total_files']} files")


if __name__ == "__main__":
    main()
