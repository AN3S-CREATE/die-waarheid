"""
Unified Analysis Engine for Die Waarheid
Integrates text forensics, audio analysis, timeline, and chat parsing
Provides single interface for complete forensic analysis

This module brings together:
- Text forensic analysis (patterns, contradictions, story flow)
- Audio forensic analysis (stress, emotion, voice patterns)
- Afrikaans verification (transcription accuracy)
- Timeline reconstruction (chronological ordering)
- Chat parsing (WhatsApp export processing)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Type of evidence source"""
    CHAT_TEXT = "chat_text"
    VOICE_NOTE = "voice_note"
    TRANSCRIPTION = "transcription"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


class EvidenceStatus(Enum):
    """Status of evidence verification"""
    VERIFIED = "verified"
    PENDING = "pending"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass
class UnifiedEntry:
    """Single entry in unified timeline with all analysis"""
    entry_id: str
    timestamp: datetime
    source_type: SourceType
    sender: str
    
    # Content
    original_text: str
    translated_text: Optional[str] = None
    file_path: Optional[str] = None
    
    # Analysis results
    text_analysis: Dict[str, Any] = field(default_factory=dict)
    audio_analysis: Dict[str, Any] = field(default_factory=dict)
    afrikaans_verification: Dict[str, Any] = field(default_factory=dict)
    
    # Forensic flags
    forensic_flags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    # Verification
    status: EvidenceStatus = EvidenceStatus.PENDING
    verification_notes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'source_type': self.source_type.value,
            'sender': self.sender,
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'file_path': self.file_path,
            'text_analysis': self.text_analysis,
            'audio_analysis': self.audio_analysis,
            'afrikaans_verification': self.afrikaans_verification,
            'forensic_flags': self.forensic_flags,
            'risk_score': self.risk_score,
            'status': self.status.value,
            'verification_notes': self.verification_notes,
            'confidence': self.confidence
        }


@dataclass
class UnifiedReport:
    """Complete unified analysis report"""
    report_id: str
    case_name: str
    generated_at: str
    
    # Statistics
    total_entries: int
    entries_by_source: Dict[str, int]
    entries_by_sender: Dict[str, int]
    date_range: Dict[str, str]
    
    # Unified timeline
    timeline: List[UnifiedEntry]
    
    # Combined analysis
    text_forensics_summary: Dict[str, Any]
    audio_forensics_summary: Dict[str, Any]
    psychological_summary: Dict[str, Any]
    
    # Risk assessment
    overall_risk: Dict[str, Any]
    critical_findings: List[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[str]


class UnifiedAnalyzer:
    """
    Main unified analysis engine
    Combines all analysis modules
    """

    def __init__(self):
        """Initialize unified analyzer"""
        self.entry_counter = 0
        self.report_counter = 0
        self.entries: List[UnifiedEntry] = []
        
        # Initialize sub-analyzers
        self._init_analyzers()

    def _init_analyzers(self):
        """Initialize all analysis modules"""
        # Text forensics
        try:
            from text_forensics import TextForensicsEngine, TextMessage
            self.text_forensics = TextForensicsEngine()
            self.TextMessage = TextMessage
            logger.info("Text forensics engine loaded")
        except ImportError as e:
            logger.warning(f"Text forensics not available: {e}")
            self.text_forensics = None

        # Audio forensics
        try:
            from forensics import ForensicsEngine
            self.audio_forensics = ForensicsEngine()
            logger.info("Audio forensics engine loaded")
        except ImportError as e:
            logger.warning(f"Audio forensics not available: {e}")
            self.audio_forensics = None

        # Afrikaans processor
        try:
            from afrikaans_processor import AfrikaansProcessor
            self.afrikaans_processor = AfrikaansProcessor()
            logger.info("Afrikaans processor loaded")
        except ImportError as e:
            logger.warning(f"Afrikaans processor not available: {e}")
            self.afrikaans_processor = None

        # Timeline reconstructor
        try:
            from timeline_reconstruction import TimelineReconstructor
            self.timeline_reconstructor = TimelineReconstructor()
            logger.info("Timeline reconstructor loaded")
        except ImportError as e:
            logger.warning(f"Timeline reconstructor not available: {e}")
            self.timeline_reconstructor = None

        # Chat parser
        try:
            from chat_parser import ChatParser
            self.chat_parser = ChatParser()
            logger.info("Chat parser loaded")
        except ImportError as e:
            logger.warning(f"Chat parser not available: {e}")
            self.chat_parser = None

        # AI Analyzer
        try:
            from ai_analyzer import AIAnalyzer
            self.ai_analyzer = AIAnalyzer()
            logger.info("AI analyzer loaded")
        except ImportError as e:
            logger.warning(f"AI analyzer not available: {e}")
            self.ai_analyzer = None

    def load_chat_export(self, file_path: Path) -> int:
        """
        Load and parse WhatsApp chat export

        Args:
            file_path: Path to chat export file

        Returns:
            Number of entries loaded
        """
        if not self.chat_parser:
            logger.error("Chat parser not available")
            return 0

        try:
            messages = self.chat_parser.parse_file(str(file_path))
            
            for msg in messages:
                self.entry_counter += 1
                
                entry = UnifiedEntry(
                    entry_id=f"UE_{self.entry_counter:06d}",
                    timestamp=msg.get('timestamp', datetime.now()),
                    source_type=SourceType.CHAT_TEXT,
                    sender=msg.get('sender', 'Unknown'),
                    original_text=msg.get('message', ''),
                    file_path=str(file_path)
                )
                
                # Check if media message
                if '<Media omitted>' in entry.original_text or entry.original_text.startswith('['):
                    entry.source_type = SourceType.VOICE_NOTE  # Could be voice note
                
                self.entries.append(entry)
            
            logger.info(f"Loaded {len(messages)} messages from chat export")
            return len(messages)

        except Exception as e:
            logger.error(f"Error loading chat export: {e}")
            return 0

    def load_voice_notes(self, directory: Path) -> int:
        """
        Load voice notes from directory

        Args:
            directory: Directory containing voice notes

        Returns:
            Number of voice notes loaded
        """
        if not self.timeline_reconstructor:
            logger.error("Timeline reconstructor not available")
            return 0

        try:
            entries = self.timeline_reconstructor.add_files_from_directory(
                directory,
                extensions=['.opus', '.ogg', '.mp3', '.m4a', '.wav']
            )
            
            for tl_entry in entries:
                self.entry_counter += 1
                
                entry = UnifiedEntry(
                    entry_id=f"UE_{self.entry_counter:06d}",
                    timestamp=tl_entry.timestamp,
                    source_type=SourceType.VOICE_NOTE,
                    sender=tl_entry.speaker_id or 'Unknown',
                    original_text=tl_entry.transcription or '',
                    translated_text=tl_entry.translation,
                    file_path=tl_entry.file_path
                )
                
                self.entries.append(entry)
            
            logger.info(f"Loaded {len(entries)} voice notes")
            return len(entries)

        except Exception as e:
            logger.error(f"Error loading voice notes: {e}")
            return 0

    def transcribe_voice_notes(self, use_afrikaans: bool = True) -> int:
        """
        Transcribe all voice notes

        Args:
            use_afrikaans: Whether to use Afrikaans verification

        Returns:
            Number of notes transcribed
        """
        transcribed = 0
        
        voice_entries = [e for e in self.entries if e.source_type == SourceType.VOICE_NOTE]
        
        for entry in voice_entries:
            if not entry.file_path or entry.original_text:
                continue  # Already has text or no file

            try:
                # Transcribe with Whisper via audio forensics
                if self.audio_forensics:
                    result = self.audio_forensics.transcribe_audio(Path(entry.file_path))
                    if result:
                        entry.original_text = result.get('text', '')
                        entry.audio_analysis = result

                # Verify Afrikaans if enabled
                if use_afrikaans and self.afrikaans_processor and entry.original_text:
                    verification = self.afrikaans_processor.process_text(
                        entry.original_text,
                        speaker_id=entry.sender,
                        audio_layer="foreground"
                    )
                    
                    entry.translated_text = verification.translated_english
                    entry.afrikaans_verification = {
                        'confidence': verification.overall_confidence,
                        'confidence_level': verification.overall_confidence_level.value,
                        'requires_review': verification.requires_human_review,
                        'all_checks_passed': verification.all_checks_passed
                    }
                    
                    if verification.requires_human_review:
                        entry.forensic_flags.append("AFRIKAANS_REVIEW_REQUIRED")
                        entry.status = EvidenceStatus.FLAGGED

                transcribed += 1
                entry.source_type = SourceType.TRANSCRIPTION

            except Exception as e:
                logger.error(f"Error transcribing {entry.file_path}: {e}")
                entry.forensic_flags.append(f"TRANSCRIPTION_ERROR: {str(e)}")

        logger.info(f"Transcribed {transcribed} voice notes")
        return transcribed

    def run_text_forensics(self) -> Dict[str, Any]:
        """
        Run text forensic analysis on all text entries

        Returns:
            Text forensics summary
        """
        if not self.text_forensics:
            return {'error': 'Text forensics not available'}

        try:
            # Convert entries to TextMessage format
            text_messages = []
            for entry in self.entries:
                if entry.original_text:
                    text_messages.append(self.TextMessage(
                        message_id=entry.entry_id,
                        timestamp=entry.timestamp,
                        sender=entry.sender,
                        text=entry.original_text
                    ))

            if not text_messages:
                return {'error': 'No text messages to analyze'}

            # Run analysis
            report = self.text_forensics.analyze_conversation(text_messages)
            
            # Update entries with findings
            for finding in report.findings:
                for msg_id in finding.message_ids:
                    for entry in self.entries:
                        if entry.entry_id == msg_id:
                            entry.forensic_flags.append(f"{finding.analysis_type.value}:{finding.title}")
                            entry.risk_score = max(entry.risk_score, finding.confidence * 100)
                            
                            if finding.severity.value in ['critical', 'high']:
                                entry.status = EvidenceStatus.FLAGGED

            return report.to_dict()

        except Exception as e:
            logger.error(f"Text forensics error: {e}")
            return {'error': str(e)}

    def run_audio_forensics(self) -> Dict[str, Any]:
        """
        Run audio forensic analysis on voice notes

        Returns:
            Audio forensics summary
        """
        if not self.audio_forensics:
            return {'error': 'Audio forensics not available'}

        results = {
            'analyzed': 0,
            'high_stress': 0,
            'entries': []
        }

        voice_entries = [e for e in self.entries if e.file_path and 
                        e.source_type in [SourceType.VOICE_NOTE, SourceType.TRANSCRIPTION]]

        for entry in voice_entries:
            try:
                analysis = self.audio_forensics.analyze(Path(entry.file_path))
                
                if analysis:
                    entry.audio_analysis = analysis
                    results['analyzed'] += 1
                    
                    # Check stress level
                    stress = analysis.get('stress_level', 0)
                    if stress > 50:
                        results['high_stress'] += 1
                        entry.forensic_flags.append(f"HIGH_STRESS:{stress:.0f}")
                        entry.status = EvidenceStatus.FLAGGED
                    
                    results['entries'].append({
                        'entry_id': entry.entry_id,
                        'stress_level': stress,
                        'pitch_volatility': analysis.get('pitch_volatility', 0),
                        'silence_ratio': analysis.get('silence_ratio', 0)
                    })

            except Exception as e:
                logger.error(f"Audio analysis error for {entry.entry_id}: {e}")

        return results

    def run_psychological_analysis(self) -> Dict[str, Any]:
        """
        Run AI psychological analysis

        Returns:
            Psychological analysis summary
        """
        if not self.ai_analyzer:
            return {'error': 'AI analyzer not available'}

        try:
            # Prepare conversation data
            messages = []
            for entry in sorted(self.entries, key=lambda x: x.timestamp):
                if entry.original_text:
                    messages.append({
                        'sender': entry.sender,
                        'text': entry.original_text,
                        'timestamp': entry.timestamp.isoformat()
                    })

            if not messages:
                return {'error': 'No messages to analyze'}

            # Run psychological profile
            profile = self.ai_analyzer.generate_psychological_profile(messages)
            
            # Detect contradictions
            contradictions = self.ai_analyzer.detect_contradictions(messages)
            
            return {
                'psychological_profile': profile,
                'contradictions': contradictions,
                'message_count': len(messages)
            }

        except Exception as e:
            logger.error(f"Psychological analysis error: {e}")
            return {'error': str(e)}

    def get_unified_timeline(self) -> List[Dict]:
        """
        Get chronologically sorted unified timeline

        Returns:
            List of timeline entries
        """
        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp)
        return [e.to_dict() for e in sorted_entries]

    def get_flagged_entries(self) -> List[Dict]:
        """
        Get all flagged entries requiring review

        Returns:
            List of flagged entries
        """
        flagged = [e for e in self.entries if e.status == EvidenceStatus.FLAGGED]
        return [e.to_dict() for e in sorted(flagged, key=lambda x: x.risk_score, reverse=True)]

    def get_entries_by_sender(self, sender: str) -> List[Dict]:
        """
        Get all entries from specific sender

        Args:
            sender: Sender name

        Returns:
            List of entries
        """
        sender_entries = [e for e in self.entries if e.sender == sender]
        return [e.to_dict() for e in sorted(sender_entries, key=lambda x: x.timestamp)]

    def generate_unified_report(self, case_name: str = "Case Analysis") -> UnifiedReport:
        """
        Generate complete unified analysis report

        Args:
            case_name: Name for the case

        Returns:
            Complete unified report
        """
        self.report_counter += 1
        report_id = f"UR_{self.report_counter:06d}"
        
        logger.info(f"Generating unified report: {report_id}")

        # Run all analyses
        text_summary = self.run_text_forensics()
        audio_summary = self.run_audio_forensics()
        psych_summary = self.run_psychological_analysis()

        # Calculate statistics
        by_source = {}
        by_sender = {}
        
        for entry in self.entries:
            src = entry.source_type.value
            by_source[src] = by_source.get(src, 0) + 1
            by_sender[entry.sender] = by_sender.get(entry.sender, 0) + 1

        # Date range
        timestamps = [e.timestamp for e in self.entries]
        date_range = {
            'start': min(timestamps).isoformat() if timestamps else '',
            'end': max(timestamps).isoformat() if timestamps else ''
        }

        # Critical findings
        critical_findings = []
        flagged = [e for e in self.entries if e.status == EvidenceStatus.FLAGGED]
        for entry in sorted(flagged, key=lambda x: x.risk_score, reverse=True)[:10]:
            critical_findings.append({
                'entry_id': entry.entry_id,
                'timestamp': entry.timestamp.isoformat(),
                'sender': entry.sender,
                'flags': entry.forensic_flags,
                'risk_score': entry.risk_score
            })

        # Overall risk assessment
        avg_risk = sum(e.risk_score for e in self.entries) / max(len(self.entries), 1)
        flagged_count = len(flagged)
        
        if avg_risk > 70 or flagged_count > len(self.entries) * 0.3:
            risk_level = "CRITICAL"
        elif avg_risk > 50 or flagged_count > len(self.entries) * 0.2:
            risk_level = "HIGH"
        elif avg_risk > 30 or flagged_count > len(self.entries) * 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        overall_risk = {
            'level': risk_level,
            'average_risk_score': avg_risk,
            'flagged_entries': flagged_count,
            'total_entries': len(self.entries)
        }

        # Recommendations
        recommendations = self._generate_recommendations(
            text_summary, audio_summary, psych_summary, overall_risk
        )

        return UnifiedReport(
            report_id=report_id,
            case_name=case_name,
            generated_at=datetime.now().isoformat(),
            total_entries=len(self.entries),
            entries_by_source=by_source,
            entries_by_sender=by_sender,
            date_range=date_range,
            timeline=self.entries,
            text_forensics_summary=text_summary,
            audio_forensics_summary=audio_summary,
            psychological_summary=psych_summary,
            overall_risk=overall_risk,
            critical_findings=critical_findings,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        text_summary: Dict,
        audio_summary: Dict,
        psych_summary: Dict,
        risk: Dict
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Risk-based recommendations
        if risk.get('level') == 'CRITICAL':
            recommendations.append("⚠️ CRITICAL: Immediate review of all flagged communications required")
        elif risk.get('level') == 'HIGH':
            recommendations.append("⚠️ HIGH RISK: Priority review of flagged entries recommended")

        # Text forensics recommendations
        if text_summary.get('summary', {}).get('critical_count', 0) > 0:
            recommendations.append("Critical text forensic findings detected - review contradictions and patterns")

        # Audio recommendations
        if audio_summary.get('high_stress', 0) > 0:
            recommendations.append(f"High stress detected in {audio_summary['high_stress']} voice notes - correlate with timeline")

        # Psychological recommendations
        if psych_summary.get('psychological_profile'):
            recommendations.append("Complete psychological profile available - review for manipulation indicators")

        # Afrikaans recommendations
        afrikaans_flagged = sum(1 for e in self.entries if 'AFRIKAANS_REVIEW_REQUIRED' in e.forensic_flags)
        if afrikaans_flagged > 0:
            recommendations.append(f"⚠️ {afrikaans_flagged} Afrikaans transcriptions require human verification")

        if not recommendations:
            recommendations.append("No critical issues detected - standard review recommended")

        return recommendations

    def export_report(self, report: UnifiedReport, output_path: Path, format: str = 'json') -> bool:
        """
        Export report to file

        Args:
            report: Report to export
            output_path: Output file path
            format: Export format (json, html, markdown)

        Returns:
            Success status
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == 'json':
                data = {
                    'report_id': report.report_id,
                    'case_name': report.case_name,
                    'generated_at': report.generated_at,
                    'total_entries': report.total_entries,
                    'entries_by_source': report.entries_by_source,
                    'entries_by_sender': report.entries_by_sender,
                    'date_range': report.date_range,
                    'timeline': [e.to_dict() for e in report.timeline],
                    'text_forensics_summary': report.text_forensics_summary,
                    'audio_forensics_summary': report.audio_forensics_summary,
                    'psychological_summary': report.psychological_summary,
                    'overall_risk': report.overall_risk,
                    'critical_findings': report.critical_findings,
                    'recommendations': report.recommendations
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

            elif format == 'markdown':
                md = self._generate_markdown_report(report)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(md)

            elif format == 'html':
                html = self._generate_html_report(report)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)

            logger.info(f"Exported report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False

    def _generate_markdown_report(self, report: UnifiedReport) -> str:
        """Generate Markdown report"""
        lines = [
            f"# {report.case_name}",
            f"\n**Report ID:** {report.report_id}",
            f"**Generated:** {report.generated_at}",
            f"\n## Overview",
            f"- **Total Entries:** {report.total_entries}",
            f"- **Date Range:** {report.date_range.get('start', '')} to {report.date_range.get('end', '')}",
            f"- **Risk Level:** {report.overall_risk.get('level', 'Unknown')}",
            f"\n## Entries by Source",
        ]
        
        for src, count in report.entries_by_source.items():
            lines.append(f"- {src}: {count}")

        lines.append("\n## Entries by Sender")
        for sender, count in report.entries_by_sender.items():
            lines.append(f"- {sender}: {count}")

        lines.append("\n## Risk Assessment")
        lines.append(f"- **Level:** {report.overall_risk.get('level', 'Unknown')}")
        lines.append(f"- **Average Risk Score:** {report.overall_risk.get('average_risk_score', 0):.1f}")
        lines.append(f"- **Flagged Entries:** {report.overall_risk.get('flagged_entries', 0)}")

        lines.append("\n## Critical Findings")
        for finding in report.critical_findings:
            lines.append(f"\n### {finding.get('entry_id', '')}")
            lines.append(f"- **Timestamp:** {finding.get('timestamp', '')}")
            lines.append(f"- **Sender:** {finding.get('sender', '')}")
            lines.append(f"- **Risk Score:** {finding.get('risk_score', 0):.1f}")
            lines.append(f"- **Flags:** {', '.join(finding.get('flags', []))}")

        lines.append("\n## Recommendations")
        for rec in report.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def _generate_html_report(self, report: UnifiedReport) -> str:
        """Generate HTML report"""
        risk_colors = {
            'CRITICAL': '#e74c3c',
            'HIGH': '#e67e22',
            'MEDIUM': '#f1c40f',
            'LOW': '#27ae60'
        }
        
        risk_level = report.overall_risk.get('level', 'LOW')
        risk_color = risk_colors.get(risk_level, '#95a5a6')

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.case_name} - Forensic Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .risk-badge {{ display: inline-block; padding: 10px 20px; border-radius: 5px; color: white; font-weight: bold; background: {risk_color}; }}
        .stat-box {{ display: inline-block; background: #ecf0f1; padding: 15px 25px; margin: 5px; border-radius: 5px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; }}
        .finding {{ background: #fff5f5; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }}
        .recommendation {{ background: #f0fff4; border-left: 4px solid #27ae60; padding: 10px 15px; margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 {report.case_name}</h1>
        <p><strong>Report ID:</strong> {report.report_id}<br>
        <strong>Generated:</strong> {report.generated_at}</p>
        
        <h2>Risk Assessment</h2>
        <span class="risk-badge">{risk_level} RISK</span>
        
        <h2>Overview Statistics</h2>
        <div class="stat-box">
            <div class="stat-value">{report.total_entries}</div>
            <div class="stat-label">Total Entries</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{report.overall_risk.get('flagged_entries', 0)}</div>
            <div class="stat-label">Flagged</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{report.overall_risk.get('average_risk_score', 0):.0f}</div>
            <div class="stat-label">Avg Risk Score</div>
        </div>
        
        <h2>Entries by Source</h2>
        <table>
            <tr><th>Source</th><th>Count</th></tr>
"""
        
        for src, count in report.entries_by_source.items():
            html += f"            <tr><td>{src}</td><td>{count}</td></tr>\n"
        
        html += """        </table>
        
        <h2>Entries by Sender</h2>
        <table>
            <tr><th>Sender</th><th>Count</th></tr>
"""
        
        for sender, count in report.entries_by_sender.items():
            html += f"            <tr><td>{sender}</td><td>{count}</td></tr>\n"
        
        html += """        </table>
        
        <h2>Critical Findings</h2>
"""
        
        for finding in report.critical_findings:
            html += f"""        <div class="finding">
            <strong>{finding.get('entry_id', '')}</strong> - {finding.get('sender', '')} ({finding.get('timestamp', '')})<br>
            Risk Score: {finding.get('risk_score', 0):.0f}<br>
            Flags: {', '.join(finding.get('flags', []))}
        </div>
"""
        
        html += """        <h2>Recommendations</h2>
"""
        
        for rec in report.recommendations:
            html += f'        <div class="recommendation">{rec}</div>\n'
        
        html += """    </div>
</body>
</html>"""
        
        return html


if __name__ == "__main__":
    analyzer = UnifiedAnalyzer()
    print("Unified Analyzer initialized")
    print("\nUsage:")
    print("  analyzer.load_chat_export(Path('chat.txt'))")
    print("  analyzer.load_voice_notes(Path('voice_notes/'))")
    print("  analyzer.transcribe_voice_notes(use_afrikaans=True)")
    print("  report = analyzer.generate_unified_report('Case Name')")
    print("  analyzer.export_report(report, Path('report.html'), format='html')")
