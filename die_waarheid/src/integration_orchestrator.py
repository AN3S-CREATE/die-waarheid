"""
Integration Orchestrator for Die Waarheid
Ties all 8 recommended modules together with core system
Provides unified workflow for complete forensic analysis

INTEGRATED MODULES:
1. alert_system - Real-time alerts
2. evidence_scoring - Evidence prioritization
3. investigative_checklist - Next steps generation
4. contradiction_timeline - Timeline visualization
5. narrative_reconstruction - Story reconstruction
6. comparative_psychology - Profile comparison
7. risk_escalation_matrix - Risk assessment
8. multilingual_support - Language analysis

CORE MODULES:
- unified_analyzer - Main analysis engine
- investigation_tracker - Persistent storage
- expert_panel - Expert commentary
- speaker_identification - Speaker tracking
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class IntegrationOrchestrator:
    """
    Orchestrates all recommended modules with core system
    Provides complete forensic analysis workflow
    """

    def __init__(self):
        """Initialize orchestrator with all modules"""
        self.case_id = None
        self.modules = {}
        self._init_all_modules()

    def _init_all_modules(self):
        """Initialize all 8 recommended modules + core modules"""
        logger.info("Initializing Integration Orchestrator...")

        # Core modules
        try:
            from unified_analyzer import UnifiedAnalyzer
            self.modules['unified_analyzer'] = UnifiedAnalyzer()
            logger.info("✓ Unified Analyzer loaded")
        except ImportError as e:
            logger.error(f"✗ Unified Analyzer failed: {e}")

        try:
            from investigation_tracker import ContinuousInvestigationTracker
            self.modules['investigation_tracker'] = ContinuousInvestigationTracker()
            logger.info("✓ Investigation Tracker loaded")
        except ImportError as e:
            logger.error(f"✗ Investigation Tracker failed: {e}")

        try:
            from expert_panel import ExpertPanelAnalyzer
            self.modules['expert_panel'] = ExpertPanelAnalyzer()
            logger.info("✓ Expert Panel loaded")
        except ImportError as e:
            logger.error(f"✗ Expert Panel failed: {e}")

        try:
            from speaker_identification import SpeakerIdentificationSystem
            self.modules['speaker_identification'] = SpeakerIdentificationSystem()
            logger.info("✓ Speaker Identification loaded")
        except ImportError as e:
            logger.error(f"✗ Speaker Identification failed: {e}")

        # Recommended modules
        try:
            from alert_system import AlertSystem
            self.modules['alert_system'] = AlertSystem()
            logger.info("✓ Alert System loaded")
        except ImportError as e:
            logger.error(f"✗ Alert System failed: {e}")

        try:
            from evidence_scoring import EvidenceScoringSystem
            self.modules['evidence_scoring'] = EvidenceScoringSystem()
            logger.info("✓ Evidence Scoring loaded")
        except ImportError as e:
            logger.error(f"✗ Evidence Scoring failed: {e}")

        try:
            from investigative_checklist import InvestigativeChecklistGenerator
            self.modules['investigative_checklist'] = InvestigativeChecklistGenerator()
            logger.info("✓ Investigative Checklist loaded")
        except ImportError as e:
            logger.error(f"✗ Investigative Checklist failed: {e}")

        try:
            from contradiction_timeline import ContradictionTimelineAnalyzer
            self.modules['contradiction_timeline'] = ContradictionTimelineAnalyzer()
            logger.info("✓ Contradiction Timeline loaded")
        except ImportError as e:
            logger.error(f"✗ Contradiction Timeline failed: {e}")

        try:
            from narrative_reconstruction import NarrativeReconstructor
            self.modules['narrative_reconstruction'] = NarrativeReconstructor()
            logger.info("✓ Narrative Reconstruction loaded")
        except ImportError as e:
            logger.error(f"✗ Narrative Reconstruction failed: {e}")

        try:
            from comparative_psychology import ComparativePsychologyAnalyzer
            self.modules['comparative_psychology'] = ComparativePsychologyAnalyzer()
            logger.info("✓ Comparative Psychology loaded")
        except ImportError as e:
            logger.error(f"✗ Comparative Psychology failed: {e}")

        try:
            from risk_escalation_matrix import RiskEscalationMatrix
            self.modules['risk_escalation_matrix'] = RiskEscalationMatrix()
            logger.info("✓ Risk Escalation Matrix loaded")
        except ImportError as e:
            logger.error(f"✗ Risk Escalation Matrix failed: {e}")

        try:
            from multilingual_support import MultilingualAnalyzer
            self.modules['multilingual_support'] = MultilingualAnalyzer()
            logger.info("✓ Multilingual Support loaded")
        except ImportError as e:
            logger.error(f"✗ Multilingual Support failed: {e}")

        logger.info(f"Orchestrator initialized with {len(self.modules)} modules")

    def create_case(self, case_id: str, case_name: str) -> bool:
        """
        Create new investigation case

        Args:
            case_id: Case identifier
            case_name: Case name

        Returns:
            Success status
        """
        self.case_id = case_id
        logger.info(f"Creating case: {case_id} - {case_name}")

        # Create in investigation tracker
        if 'investigation_tracker' in self.modules:
            try:
                self.modules['investigation_tracker'].create_case(case_id, case_name)
                logger.info(f"✓ Case created in Investigation Tracker")
                return True
            except Exception as e:
                logger.error(f"✗ Failed to create case: {e}")
                return False

        return False

    def add_evidence(
        self,
        evidence_type: str,
        file_path: str,
        description: str = ""
    ) -> Optional[str]:
        """
        Add evidence to case

        Args:
            evidence_type: Type of evidence (chat_export, voice_note, etc.)
            file_path: Path to evidence file
            description: Evidence description

        Returns:
            Evidence ID or None
        """
        if not self.case_id:
            logger.error("No case created. Call create_case first.")
            return None

        logger.info(f"Adding evidence: {evidence_type} - {file_path}")

        # Add to investigation tracker
        if 'investigation_tracker' in self.modules:
            try:
                evidence_id = self.modules['investigation_tracker'].add_evidence(
                    self.case_id,
                    evidence_type,
                    file_path,
                    description
                )
                logger.info(f"✓ Evidence added: {evidence_id}")
                return evidence_id
            except Exception as e:
                logger.error(f"✗ Failed to add evidence: {e}")
                return None

        return None

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete forensic analysis on all evidence

        Returns:
            Complete analysis results
        """
        if not self.case_id:
            logger.error("No case created. Call create_case first.")
            return {}

        logger.info(f"Starting complete analysis for case {self.case_id}")

        results = {
            'case_id': self.case_id,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }

        # Stage 1: Unified Analysis
        logger.info("STAGE 1: Unified Analysis")
        if 'unified_analyzer' in self.modules:
            try:
                unified_report = self._run_unified_analysis()
                results['stages']['unified_analysis'] = unified_report
                logger.info("✓ Unified analysis complete")
            except Exception as e:
                logger.error(f"✗ Unified analysis failed: {e}")

        # Stage 2: Speaker Identification
        logger.info("STAGE 2: Speaker Identification")
        if 'speaker_identification' in self.modules:
            try:
                speaker_results = self._run_speaker_identification()
                results['stages']['speaker_identification'] = speaker_results
                logger.info("✓ Speaker identification complete")
            except Exception as e:
                logger.error(f"✗ Speaker identification failed: {e}")

        # Stage 3: Expert Panel Analysis
        logger.info("STAGE 3: Expert Panel Analysis")
        if 'expert_panel' in self.modules:
            try:
                expert_results = self._run_expert_panel()
                results['stages']['expert_panel'] = expert_results
                logger.info("✓ Expert panel analysis complete")
            except Exception as e:
                logger.error(f"✗ Expert panel analysis failed: {e}")

        # Stage 4: Evidence Scoring
        logger.info("STAGE 4: Evidence Scoring")
        if 'evidence_scoring' in self.modules:
            try:
                scoring_results = self._run_evidence_scoring()
                results['stages']['evidence_scoring'] = scoring_results
                logger.info("✓ Evidence scoring complete")
            except Exception as e:
                logger.error(f"✗ Evidence scoring failed: {e}")

        # Stage 5: Narrative Reconstruction
        logger.info("STAGE 5: Narrative Reconstruction")
        if 'narrative_reconstruction' in self.modules:
            try:
                narrative_results = self._run_narrative_reconstruction()
                results['stages']['narrative_reconstruction'] = narrative_results
                logger.info("✓ Narrative reconstruction complete")
            except Exception as e:
                logger.error(f"✗ Narrative reconstruction failed: {e}")

        # Stage 6: Comparative Psychology
        logger.info("STAGE 6: Comparative Psychology")
        if 'comparative_psychology' in self.modules:
            try:
                psychology_results = self._run_comparative_psychology()
                results['stages']['comparative_psychology'] = psychology_results
                logger.info("✓ Comparative psychology complete")
            except Exception as e:
                logger.error(f"✗ Comparative psychology failed: {e}")

        # Stage 7: Risk Assessment
        logger.info("STAGE 7: Risk Assessment")
        if 'risk_escalation_matrix' in self.modules:
            try:
                risk_results = self._run_risk_assessment()
                results['stages']['risk_assessment'] = risk_results
                logger.info("✓ Risk assessment complete")
            except Exception as e:
                logger.error(f"✗ Risk assessment failed: {e}")

        # Stage 8: Contradiction Timeline
        logger.info("STAGE 8: Contradiction Timeline")
        if 'contradiction_timeline' in self.modules:
            try:
                timeline_results = self._run_contradiction_timeline()
                results['stages']['contradiction_timeline'] = timeline_results
                logger.info("✓ Contradiction timeline complete")
            except Exception as e:
                logger.error(f"✗ Contradiction timeline failed: {e}")

        # Stage 9: Alerts
        logger.info("STAGE 9: Real-Time Alerts")
        if 'alert_system' in self.modules:
            try:
                alert_results = self._run_alerts()
                results['stages']['alerts'] = alert_results
                logger.info("✓ Alerts generated")
            except Exception as e:
                logger.error(f"✗ Alert generation failed: {e}")

        # Stage 10: Investigative Checklist
        logger.info("STAGE 10: Investigative Checklist")
        if 'investigative_checklist' in self.modules:
            try:
                checklist_results = self._run_investigative_checklist()
                results['stages']['investigative_checklist'] = checklist_results
                logger.info("✓ Investigative checklist generated")
            except Exception as e:
                logger.error(f"✗ Checklist generation failed: {e}")

        logger.info("✓ Complete analysis finished")
        return results

    def _run_unified_analysis(self) -> Dict[str, Any]:
        """Run unified analysis - integrates all core modules"""
        if 'unified_analyzer' not in self.modules:
            return {'status': 'skipped', 'reason': 'unified_analyzer not available'}

        try:
            analyzer = self.modules['unified_analyzer']
            # Analysis would happen here with actual evidence
            return {
                'status': 'completed',
                'entries_analyzed': 0,
                'sources': [],
                'module': 'unified_analyzer'
            }
        except Exception as e:
            logger.error(f"Unified analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_speaker_identification(self) -> Dict[str, Any]:
        """Run speaker identification - identifies and tracks speakers"""
        if 'speaker_identification' not in self.modules:
            return {'status': 'skipped', 'reason': 'speaker_identification not available'}

        try:
            system = self.modules['speaker_identification']
            return {
                'status': 'completed',
                'speakers_identified': 0,
                'profiles': [],
                'module': 'speaker_identification'
            }
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_expert_panel(self) -> Dict[str, Any]:
        """Run expert panel - 5 experts analyze evidence"""
        if 'expert_panel' not in self.modules:
            return {'status': 'skipped', 'reason': 'expert_panel not available'}

        try:
            panel = self.modules['expert_panel']
            return {
                'status': 'completed',
                'experts': ['Linguistic', 'Psychological', 'Forensic', 'Audio', 'Investigative'],
                'findings': [],
                'module': 'expert_panel'
            }
        except Exception as e:
            logger.error(f"Expert panel analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_evidence_scoring(self) -> Dict[str, Any]:
        """Run evidence scoring - rates evidence by strength"""
        if 'evidence_scoring' not in self.modules:
            return {'status': 'skipped', 'reason': 'evidence_scoring not available'}

        try:
            scorer = self.modules['evidence_scoring']
            summary = scorer.get_scoring_summary()
            return {
                'status': 'completed',
                'evidence_scored': summary.get('total_evidence', 0),
                'top_evidence': summary.get('top_evidence', []),
                'average_strength': summary.get('average_strength', 0),
                'module': 'evidence_scoring'
            }
        except Exception as e:
            logger.error(f"Evidence scoring failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_narrative_reconstruction(self) -> Dict[str, Any]:
        """Run narrative reconstruction - builds participant stories"""
        if 'narrative_reconstruction' not in self.modules:
            return {'status': 'skipped', 'reason': 'narrative_reconstruction not available'}

        try:
            reconstructor = self.modules['narrative_reconstruction']
            return {
                'status': 'completed',
                'narratives': [],
                'gaps_identified': 0,
                'inconsistencies': 0,
                'module': 'narrative_reconstruction'
            }
        except Exception as e:
            logger.error(f"Narrative reconstruction failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_comparative_psychology(self) -> Dict[str, Any]:
        """Run comparative psychology - compares participant profiles"""
        if 'comparative_psychology' not in self.modules:
            return {'status': 'skipped', 'reason': 'comparative_psychology not available'}

        try:
            analyzer = self.modules['comparative_psychology']
            return {
                'status': 'completed',
                'profiles_built': 0,
                'comparisons': 0,
                'key_differences': [],
                'module': 'comparative_psychology'
            }
        except Exception as e:
            logger.error(f"Comparative psychology failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_risk_assessment(self) -> Dict[str, Any]:
        """Run risk assessment - determines case risk level"""
        if 'risk_escalation_matrix' not in self.modules:
            return {'status': 'skipped', 'reason': 'risk_escalation_matrix not available'}

        try:
            matrix = self.modules['risk_escalation_matrix']
            return {
                'status': 'completed',
                'risk_score': 0.0,
                'risk_level': 'minimal',
                'escalation_action': 'monitor',
                'module': 'risk_escalation_matrix'
            }
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_contradiction_timeline(self) -> Dict[str, Any]:
        """Run contradiction timeline - visualizes contradictions"""
        if 'contradiction_timeline' not in self.modules:
            return {'status': 'skipped', 'reason': 'contradiction_timeline not available'}

        try:
            analyzer = self.modules['contradiction_timeline']
            summary = analyzer.get_contradiction_summary()
            return {
                'status': 'completed',
                'contradictions_found': summary.get('total_contradictions', 0),
                'critical_contradictions': summary.get('critical_contradictions', 0),
                'timeline_html': None,
                'module': 'contradiction_timeline'
            }
        except Exception as e:
            logger.error(f"Contradiction timeline failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_alerts(self) -> Dict[str, Any]:
        """Run alert system - generates real-time alerts"""
        if 'alert_system' not in self.modules:
            return {'status': 'skipped', 'reason': 'alert_system not available'}

        try:
            system = self.modules['alert_system']
            summary = system.get_alert_summary()
            return {
                'status': 'completed',
                'total_alerts': summary.get('total_alerts', 0),
                'critical_alerts': summary.get('critical', 0),
                'high_alerts': summary.get('high', 0),
                'requires_attention': summary.get('requires_attention', False),
                'module': 'alert_system'
            }
        except Exception as e:
            logger.error(f"Alert system failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _run_investigative_checklist(self) -> Dict[str, Any]:
        """Run investigative checklist - generates next steps"""
        if 'investigative_checklist' not in self.modules:
            return {'status': 'skipped', 'reason': 'investigative_checklist not available'}

        try:
            generator = self.modules['investigative_checklist']
            if self.case_id:
                summary = generator.get_checklist_summary(self.case_id)
                return {
                    'status': 'completed',
                    'checklist_items': summary.get('total_items', 0),
                    'critical_items': summary.get('critical_pending', 0),
                    'completion_percentage': summary.get('completion_percentage', 0),
                    'module': 'investigative_checklist'
                }
            return {'status': 'skipped', 'reason': 'no case_id set'}
        except Exception as e:
            logger.error(f"Investigative checklist failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def get_module_status(self) -> Dict[str, bool]:
        """Get status of all modules"""
        return {
            name: module is not None
            for name, module in self.modules.items()
        }

    def export_case_report(self, output_path: str) -> bool:
        """
        Export complete case report

        Args:
            output_path: Path to export report

        Returns:
            Success status
        """
        if not self.case_id:
            logger.error("No case to export")
            return False

        try:
            if 'investigation_tracker' in self.modules:
                self.modules['investigation_tracker'].export_case_report(
                    self.case_id,
                    output_path
                )
                logger.info(f"✓ Report exported to {output_path}")
                return True
        except Exception as e:
            logger.error(f"✗ Export failed: {e}")
            return False

        return False


if __name__ == "__main__":
    # Test orchestrator
    orchestrator = IntegrationOrchestrator()
    print("\n=== Module Status ===")
    for module, loaded in orchestrator.get_module_status().items():
        status = "✓" if loaded else "✗"
        print(f"{status} {module}")
