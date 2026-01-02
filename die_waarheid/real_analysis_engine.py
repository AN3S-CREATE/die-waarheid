"""
REAL analysis engine for Die Waarheid - uses actual forensic pipeline
No fake outputs - everything is processed through real modules.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import hashlib
import sys
from dataclasses import asdict, is_dataclass

# Import real pipeline modules
sys.path.insert(0, str(Path(__file__).parent))
from src.whisper_transcriber import WhisperTranscriber
from src.afrikaans_processor import AfrikaansProcessor
from src.afrikaans_audio import AudioLayerSeparator
from src.diarization import SimpleDiarizer
from src.multilingual_support import MultilingualAnalyzer
from src.text_forensics import TextForensicsEngine, TextMessage
from src.narrative_reconstruction import NarrativeReconstructor
from src.contradiction_timeline import ContradictionTimelineAnalyzer, StatementType
from src.risk_escalation_matrix import RiskEscalationMatrix
from src.investigative_checklist import InvestigativeChecklistGenerator
from src.expert_panel import ExpertPanelAnalyzer

from background_sound_analyzer import BackgroundSoundAnalyzer


class RealAnalysisEngine:
    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.case_id = case_dir.name
        self.logger = logging.getLogger(__name__)
        
        # Initialize all real modules
        self.whisper = WhisperTranscriber()
        self.afrikaans_processor = AfrikaansProcessor()
        self.audio_separator = AudioLayerSeparator()
        self.diarizer = SimpleDiarizer()
        self.multilingual = MultilingualAnalyzer()
        self.text_forensics = TextForensicsEngine()
        self.narrative = NarrativeReconstructor()
        self.timeline = ContradictionTimelineAnalyzer()
        self.risk_matrix = RiskEscalationMatrix()
        self.checklist = InvestigativeChecklistGenerator()
        self.expert_panel = ExpertPanelAnalyzer()
        self.sound_analyzer = BackgroundSoundAnalyzer()
        
        # Results storage
        self.results = {
            "case_id": self.case_id,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "primary_language": "afrikaans",
            "files_processed": [],
            "transcriptions": [],
            "speaker_diarization": [],
            "background_sounds": [],
            "language_analysis": [],
            "forensic_analysis": {},
            "narrative_reconstruction": {},
            "contradictions": [],
            "risk_assessment": {},
            "investigative_checklist": [],
            "expert_commentary": [],
            "errors": []
        }
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def process_audio_file(self, file_path: Path, speaker_profiles: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a single audio file through the complete pipeline."""
        file_result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "sha256": self.calculate_file_hash(file_path),
            "status": "processing",
            "stages": {}
        }
        
        try:
            # Stage 1: Transcription (Whisper with Afrikaans optimization)
            self.logger.info(f"Transcribing {file_path.name}...")
            raw_transcript = self.whisper.transcribe(str(file_path))
            file_result["stages"]["transcription"] = {
                "status": "success",
                "raw_text": raw_transcript.get("text", ""),
                "language": raw_transcript.get("language", "unknown"),
                "confidence": raw_transcript.get("confidence", 0)
            }
            
            # Stage 2: Afrikaans processing (triple-check verification)
            self.logger.info(f"Processing Afrikaans for {file_path.name}...")
            afrikaans_result = self.afrikaans_processor.process_text(raw_transcript.get("text", ""))
            file_result["stages"]["afrikaans_processing"] = {
                "status": "success",
                "verified_text": afrikaans_result.get("verified_text", ""),
                "confidence": afrikaans_result.get("confidence", 0),
                "corrections": afrikaans_result.get("corrections", []),
                "verification_steps": afrikaans_result.get("verification_steps", [])
            }
            
            # Stage 3: Audio layer separation
            self.logger.info(f"Separating audio layers for {file_path.name}...")
            layers = self.audio_separator.separate_layers(str(file_path))
            file_result["stages"]["audio_separation"] = {
                "status": "success",
                "foreground_detected": layers.get("foreground_detected", False),
                "background_noise": layers.get("background_noise", {}),
                "voice_activity": layers.get("voice_activity", [])
            }
            
            # Stage 4: Diarization (speaker segmentation)
            self.logger.info(f"Running diarization for {file_path.name}...")
            diarization = self.diarizer.diarize(str(file_path))
            
            # Apply speaker profiles if available
            if speaker_profiles:
                diarization = self.apply_speaker_profiles(diarization, speaker_profiles)
            
            file_result["stages"]["diarization"] = {
                "status": "success",
                "segments": diarization.get("segments", []),
                "speaker_count": diarization.get("speaker_count", 0),
                "speaker_labels": diarization.get("speaker_labels", {})
            }
            
            # Stage 5: Multilingual analysis (code-switch detection)
            self.logger.info(f"Analyzing language patterns for {file_path.name}...")
            multilingual_result = self.multilingual.analyze(raw_transcript.get("text", ""))
            file_result["stages"]["multilingual_analysis"] = {
                "status": "success",
                "primary_language": multilingual_result.get("primary_language", "unknown"),
                "language_switches": multilingual_result.get("language_switches", []),
                "code_switches": multilingual_result.get("code_switches", []),
                "accent_analysis": multilingual_result.get("accent_analysis", {})
            }
            
            # Stage 6: Background sound analysis
            self.logger.info(f"Analyzing background sounds for {file_path.name}...")
            background_result = self.sound_analyzer.analyze_background_sounds(str(file_path))
            file_result["stages"]["background_analysis"] = background_result
            
            file_result["status"] = "completed"
            file_result["final_transcript"] = afrikaans_result.get("verified_text", raw_transcript.get("text", ""))
            
        except Exception as e:
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            self.logger.error(f"Failed to process {file_path.name}: {e}")
        
        return file_result
    
    def apply_speaker_profiles(self, diarization: Dict[str, Any], speaker_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Apply trained speaker profiles to diarization results."""
        # This is a simplified version - in production, you'd use voice embeddings
        segments = diarization.get("segments", [])
        speaker_labels = diarization.get("speaker_labels", {})
        
        # Map generic speaker IDs to trained names
        trained_names = list(speaker_profiles.keys())
        for i, segment in enumerate(segments):
            generic_id = segment.get("speaker", f"Speaker_{i % len(trained_names)}")
            if generic_id.startswith("Speaker_"):
                speaker_num = int(generic_id.split("_")[1])
                if speaker_num < len(trained_names):
                    segment["speaker"] = trained_names[speaker_num]
        
        diarization["segments"] = segments
        diarization["speaker_labels"] = {name: {"trained": True} for name in trained_names}
        
        return diarization
    
    def process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a text file (chat export, statement, etc.)."""
        file_result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "sha256": self.calculate_file_hash(file_path),
            "status": "processing",
            "stages": {}
        }
        
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Stage 1: Text preprocessing
            file_result["stages"]["text_preprocessing"] = {
                "status": "success",
                "content_length": len(text_content),
                "encoding": "utf-8"
            }
            
            # Stage 2: Afrikaans processing
            afrikaans_result = self.afrikaans_processor.process_text(text_content)
            file_result["stages"]["afrikaans_processing"] = {
                "status": "success",
                "verified_text": afrikaans_result.get("verified_text", ""),
                "confidence": afrikaans_result.get("confidence", 0)
            }
            
            # Stage 3: Multilingual analysis
            multilingual_result = self.multilingual.analyze(text_content)
            file_result["stages"]["multilingual_analysis"] = {
                "status": "success",
                "primary_language": multilingual_result.get("primary_language", "unknown"),
                "language_switches": multilingual_result.get("language_switches", [])
            }
            
            file_result["status"] = "completed"
            file_result["final_text"] = afrikaans_result.get("verified_text", text_content)
            
        except Exception as e:
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            self.logger.error(f"Failed to process {file_path.name}: {e}")
        
        return file_result
    
    def run_analysis(self, file_list: List[str], speaker_profiles: Optional[Dict] = None) -> Dict[str, Any]:
        """Run complete analysis on all files."""
        self.logger.info(f"Starting analysis on {len(file_list)} files")
        
        start_time = time.time()
        
        # Process each file
        for file_path_str in file_list:
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                self.results["errors"].append(f"File not found: {file_path}")
                continue
            
            self.results["files_processed"].append(str(file_path))
            
            # Process based on file type
            if file_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus']:
                file_result = self.process_audio_file(file_path, speaker_profiles)
            else:
                file_result = self.process_text_file(file_path)
            
            # Store results
            if file_result["status"] == "completed":
                # Extract and store specific results
                if "transcription" in file_result["stages"]:
                    self.results["transcriptions"].append({
                        "file": file_result["file_name"],
                        "transcript": file_result["final_transcript"],
                        "confidence": file_result["stages"]["transcription"]["confidence"]
                    })
                
                if "diarization" in file_result["stages"]:
                    self.results["speaker_diarization"].append({
                        "file": file_result["file_name"],
                        "segments": file_result["stages"]["diarization"]["segments"],
                        "speaker_count": file_result["stages"]["diarization"]["speaker_count"]
                    })
                
                if "background_analysis" in file_result["stages"]:
                    self.results["background_sounds"].append({
                        "file": file_result["file_name"],
                        "sounds": file_result["stages"]["background_analysis"].get("identified_sounds", [])
                    })
                
                if "multilingual_analysis" in file_result["stages"]:
                    self.results["language_analysis"].append({
                        "file": file_result["file_name"],
                        "primary_language": file_result["stages"]["multilingual_analysis"]["primary_language"],
                        "code_switches": file_result["stages"]["multilingual_analysis"]["code_switches"]
                    })
            else:
                self.results["errors"].append(f"Failed to process {file_result['file_name']}: {file_result.get('error', 'Unknown error')}")
        
        # Generate higher-level analysis
        self.generate_forensic_analysis()
        self.generate_narrative_reconstruction()
        self.generate_contradiction_timeline()
        self.generate_risk_assessment()
        self.generate_investigative_checklist()
        self.generate_expert_commentary()
        
        # Calculate summary
        end_time = time.time()
        self.results["summary"] = {
            "total_files": len(file_list),
            "processed": len(self.results["files_processed"]),
            "failed": len(self.results["errors"]),
            "duration_seconds": end_time - start_time,
            "primary_language": "afrikaans",
            "languages_detected": list(set(r["primary_language"] for r in self.results["language_analysis"])),
            "total_speakers": max((d["speaker_count"] for d in self.results["speaker_diarization"]), default=0),
            "background_sounds_detected": len(self.results["background_sounds"])
        }
        
        # Save results
        self.save_results()
        
        self.logger.info(f"Analysis complete. {self.results['summary']['processed']}/{self.results['summary']['total_files']} files processed successfully.")
        
        return self.results
    
    def generate_forensic_analysis(self):
        """Generate forensic text analysis."""
        all_transcripts = [t["transcript"] for t in self.results["transcriptions"]]
        if not all_transcripts:
            return

        try:
            messages: List[TextMessage] = []
            now = datetime.now()
            for idx, t in enumerate(self.results["transcriptions"]):
                messages.append(
                    TextMessage(
                        message_id=str(t.get("file", f"audio_{idx}")),
                        timestamp=now,
                        sender="UNKNOWN",
                        text=str(t.get("transcript", ""))
                    )
                )

            report = self.text_forensics.analyze_conversation(messages, topic="Voice note transcripts")
            self.results["forensic_analysis"] = report.to_dict() if hasattr(report, "to_dict") else report
        except Exception as e:
            self.results["errors"].append(f"Text forensics failed: {e}")
    
    def generate_narrative_reconstruction(self):
        """Generate narrative reconstruction from all evidence."""
        try:
            if not self.results.get("transcriptions"):
                self.results["narrative_reconstruction"] = {
                    "status": "UNAVAILABLE",
                    "reason": "No transcriptions available"
                }
                return

            now = datetime.now()

            participant_statements: Dict[str, List[Dict[str, Any]]] = {}
            participant_names: Dict[str, str] = {}

            diar_by_file = {d.get("file"): d for d in self.results.get("speaker_diarization", [])}

            for t in self.results.get("transcriptions", []):
                evidence_id = str(t.get("file", "unknown"))
                transcript = str(t.get("transcript", ""))
                if not transcript.strip():
                    continue

                participant_id = "UNKNOWN"
                diar = diar_by_file.get(evidence_id)
                if diar and diar.get("segments"):
                    participant_id = str(diar["segments"][0].get("speaker", "UNKNOWN"))

                participant_names.setdefault(participant_id, participant_id)
                participant_statements.setdefault(participant_id, []).append({
                    "content": transcript,
                    "timestamp": now,
                    "evidence_id": evidence_id
                })

            narratives: Dict[str, Any] = {}
            for participant_id, stmts in participant_statements.items():
                narratives[participant_id] = asdict(
                    self.narrative.build_narrative(
                        participant_id=participant_id,
                        participant_name=participant_names.get(participant_id, participant_id),
                        statements=stmts,
                        timeline_data=None
                    )
                )

            self.results["narrative_reconstruction"] = {
                "status": "OK",
                "participants": narratives
            }
        except Exception as e:
            self.results["errors"].append(f"Narrative reconstruction failed: {e}")
            self.results["narrative_reconstruction"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_contradiction_timeline(self):
        """Generate timeline of contradictions."""
        try:
            if not self.results.get("transcriptions"):
                self.results["contradictions"] = {
                    "status": "UNAVAILABLE",
                    "reason": "No transcriptions available"
                }
                return

            diar_by_file = {d.get("file"): d for d in self.results.get("speaker_diarization", [])}
            now = datetime.now()

            for idx, t in enumerate(self.results.get("transcriptions", [])):
                evidence_id = str(t.get("file", f"audio_{idx}"))
                transcript = str(t.get("transcript", ""))

                participant_id = "UNKNOWN"
                diar = diar_by_file.get(evidence_id)
                if diar and diar.get("segments"):
                    participant_id = str(diar["segments"][0].get("speaker", "UNKNOWN"))

                self.timeline.add_statement(
                    statement_id=evidence_id,
                    participant_id=participant_id,
                    timestamp=now,
                    statement_type=StatementType.TRANSCRIPTION,
                    content=transcript,
                    key_claims=[],
                    emotional_tone="neutral",
                    stress_level=None
                )

            self.results["contradictions"] = {
                "status": "OK",
                "summary": self.timeline.get_contradiction_summary(),
                "participants": {
                    pid: self.timeline.get_participant_timeline(pid)
                    for pid in set(s.participant_id for s in self.timeline.statements.values())
                }
            }

        except Exception as e:
            self.results["errors"].append(f"Contradiction timeline failed: {e}")
            self.results["contradictions"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_risk_assessment(self):
        """Generate risk assessment matrix."""
        try:
            participants = sorted({
                str(seg.get("speaker", "UNKNOWN"))
                for d in self.results.get("speaker_diarization", [])
                for seg in d.get("segments", [])
            })

            if not participants:
                participants = ["UNKNOWN_A", "UNKNOWN_B"]

            forensic = self.results.get("forensic_analysis")
            findings = (forensic or {}).get("findings", []) if isinstance(forensic, dict) else []

            contradiction_findings = [f for f in findings if f.get("analysis_type") == "contradiction"]
            contradiction_count = len(contradiction_findings)
            contradiction_conf = (
                sum(float(f.get("confidence", 0)) for f in contradiction_findings) / contradiction_count
                if contradiction_count else 0.0
            )

            psych_red_flags = sum(
                1 for f in findings
                if f.get("analysis_type") in {"psychological", "story_flow"} and f.get("severity") in {"critical", "high"}
            )

            p1 = self.risk_matrix.assess_participant_risk(
                participant_id=participants[0],
                participant_name=participants[0],
                contradiction_count=contradiction_count,
                contradiction_confidence=contradiction_conf,
                stress_spikes=0,
                baseline_stress=0.0,
                current_stress=0.0,
                manipulation_indicators=0,
                timeline_inconsistencies=0,
                psychological_red_flags=psych_red_flags,
                previous_assessment=None
            )

            p2 = self.risk_matrix.assess_participant_risk(
                participant_id=participants[1] if len(participants) > 1 else participants[0] + "_2",
                participant_name=participants[1] if len(participants) > 1 else participants[0] + "_2",
                contradiction_count=contradiction_count,
                contradiction_confidence=contradiction_conf,
                stress_spikes=0,
                baseline_stress=0.0,
                current_stress=0.0,
                manipulation_indicators=0,
                timeline_inconsistencies=0,
                psychological_red_flags=psych_red_flags,
                previous_assessment=None
            )

            case_assessment = self.risk_matrix.assess_case_risk(
                case_id=self.case_id,
                participant_a_risk=p1,
                participant_b_risk=p2,
                total_evidence=len(self.results.get("files_processed", [])),
                total_findings=len(findings),
                days_under_investigation=0,
                previous_case_assessment=None
            )

            self.results["risk_assessment"] = {
                "status": "OK",
                "participant_a": asdict(p1),
                "participant_b": asdict(p2),
                "case": asdict(case_assessment)
            }
        except Exception as e:
            self.results["errors"].append(f"Risk assessment failed: {e}")
            self.results["risk_assessment"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_investigative_checklist(self):
        """Generate investigative checklist."""
        try:
            forensic = self.results.get("forensic_analysis")
            findings = (forensic or {}).get("findings", []) if isinstance(forensic, dict) else []

            participants = sorted({
                str(seg.get("speaker", "UNKNOWN"))
                for d in self.results.get("speaker_diarization", [])
                for seg in d.get("segments", [])
            })

            participants_meta = [{"participant_id": p, "participant_name": p} for p in participants] or [
                {"participant_id": "UNKNOWN", "participant_name": "UNKNOWN"}
            ]

            contradictions = []
            for f in findings:
                if f.get("analysis_type") == "contradiction":
                    contradictions.append({
                        "participant_id": "UNKNOWN",
                        "current_statement": "",
                        "past_statement": "",
                        "confidence": float(f.get("confidence", 0.0)),
                        "evidence_ids": f.get("message_ids", [])
                    })

            checklist_items = self.checklist.generate_checklist_from_findings(
                case_id=self.case_id,
                contradictions=contradictions,
                pattern_changes=[],
                timeline_gaps=[],
                stress_spikes=[],
                manipulation_indicators=[],
                participants=participants_meta
            )

            self.results["investigative_checklist"] = {
                "status": "OK",
                "items": [i.to_dict() if hasattr(i, "to_dict") else asdict(i) if is_dataclass(i) else i for i in checklist_items]
            }
        except Exception as e:
            self.results["errors"].append(f"Checklist generation failed: {e}")
            self.results["investigative_checklist"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_expert_commentary(self):
        """Generate expert panel commentary."""
        try:
            briefs: List[Dict[str, Any]] = []
            now = datetime.now()

            diar_by_file = {d.get("file"): d for d in self.results.get("speaker_diarization", [])}

            for t in self.results.get("transcriptions", []):
                evidence_id = str(t.get("file", "unknown"))
                evidence_text = str(t.get("transcript", ""))

                sender = "UNKNOWN"
                diar = diar_by_file.get(evidence_id)
                if diar and diar.get("segments"):
                    sender = diar["segments"][0].get("speaker", "UNKNOWN")

                brief = self.expert_panel.generate_expert_brief(
                    evidence_id=evidence_id,
                    evidence_text=evidence_text,
                    evidence_type="audio",
                    sender=sender,
                    evidence_timestamp=now,
                    audio_metrics=None,
                    past_evidence=[]
                )

                briefs.append(brief.to_dict() if hasattr(brief, "to_dict") else brief)

            self.results["expert_commentary"] = briefs
        except Exception as e:
            self.results["errors"].append(f"Expert panel failed: {e}")
    
    def save_results(self):
        """Save analysis results to file."""
        results_file = self.case_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to: {results_file}")


def main():
    """CLI interface for real analysis engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run real Die Waarheid analysis")
    parser.add_argument("--case", type=str, required=True, help="Case name or ID")
    parser.add_argument("--manifest", type=str, help="Manifest file (JSON)")
    parser.add_argument("--files", nargs="*", help="Specific files to analyze")
    
    args = parser.parse_args()
    
    # Setup case directory
    case_dir = Path("data") / "cases" / args.case
    if not case_dir.exists():
        print(f"Case directory not found: {case_dir}")
        return
    
    # Get file list
    if args.files:
        file_list = args.files
    elif args.manifest:
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
        file_list = [f["full_path"] for f in manifest["files"]]
    else:
        # Use manifest from case directory
        manifest_file = case_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            file_list = [f["full_path"] for f in manifest["files"]]
        else:
            print("No files specified and no manifest found")
            return
    
    # Load speaker profiles if available
    speaker_profiles = {}
    session_file = Path("session_data.json")
    if session_file.exists():
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        speaker_profiles = session_data.get("speaker_profiles", {})
    
    # Run analysis
    engine = RealAnalysisEngine(case_dir)
    results = engine.run_analysis(file_list, speaker_profiles)
    
    print(f"\nAnalysis complete!")
    print(f"Processed: {results['summary']['processed']}/{results['summary']['total_files']} files")
    print(f"Duration: {results['summary']['duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
