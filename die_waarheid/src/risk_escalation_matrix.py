"""
Risk Escalation Matrix for Die Waarheid
Tracks overall case risk with dynamic escalation levels
Recommends investigation escalation based on findings

FEATURES:
- Individual risk scores per participant
- Combined case risk assessment
- Confidence in findings
- Recommended escalation levels
- Risk trend analysis
- Escalation triggers
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk escalation levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationAction(Enum):
    """Recommended escalation actions"""
    MONITOR = "monitor"
    INTERVIEW = "interview"
    CONFRONT = "confront"
    INVESTIGATE = "investigate"
    ARREST = "arrest"
    PROTECTIVE_ORDER = "protective_order"


@dataclass
class ParticipantRisk:
    """Risk assessment for single participant"""
    participant_id: str
    participant_name: str
    
    # Risk components
    contradiction_risk: float
    stress_risk: float
    manipulation_risk: float
    timeline_risk: float
    psychological_risk: float
    
    # Overall
    overall_risk_score: float
    risk_level: RiskLevel
    
    # Confidence
    confidence: float
    
    # Trends
    risk_trend: str
    risk_change: float
    
    # Recommendations
    recommended_action: EscalationAction
    action_rationale: str


@dataclass
class CaseRiskAssessment:
    """Overall case risk assessment"""
    case_id: str
    assessment_timestamp: str
    
    # Participant risks
    participant_a_risk: ParticipantRisk
    participant_b_risk: ParticipantRisk
    
    # Combined risk
    combined_risk_score: float
    case_risk_level: RiskLevel
    
    # Confidence
    overall_confidence: float
    
    # Trends
    case_risk_trend: str
    case_risk_change: float
    
    # Critical factors
    critical_factors: List[str]
    escalation_triggers: List[str]
    
    # Recommendations
    recommended_escalation: EscalationAction
    recommended_next_steps: List[str]
    
    # Timeline
    days_under_investigation: int
    evidence_count: int
    findings_count: int


class RiskEscalationMatrix:
    """
    Manages risk assessment and escalation
    """

    def __init__(self):
        """Initialize matrix"""
        self.assessments: List[CaseRiskAssessment] = []
        self.participant_risks: Dict[str, List[ParticipantRisk]] = {}
        self.escalation_history: List[Dict[str, Any]] = []

    def assess_participant_risk(
        self,
        participant_id: str,
        participant_name: str,
        contradiction_count: int,
        contradiction_confidence: float,
        stress_spikes: int,
        baseline_stress: float,
        current_stress: float,
        manipulation_indicators: int,
        timeline_inconsistencies: int,
        psychological_red_flags: int,
        previous_assessment: Optional[ParticipantRisk] = None
    ) -> ParticipantRisk:
        """
        Assess risk for single participant

        Args:
            participant_id: Participant ID
            participant_name: Participant name
            contradiction_count: Number of contradictions found
            contradiction_confidence: Average confidence of contradictions
            stress_spikes: Number of stress spikes
            baseline_stress: Baseline stress level
            current_stress: Current stress level
            manipulation_indicators: Number of manipulation indicators
            timeline_inconsistencies: Number of timeline issues
            psychological_red_flags: Number of psychological red flags
            previous_assessment: Previous risk assessment for trend

        Returns:
            Participant risk assessment
        """
        logger.info(f"Assessing risk for {participant_name}")

        # Calculate component risks (0-100)
        contradiction_risk = min(100, contradiction_count * 10 * contradiction_confidence)
        
        stress_risk = 0.0
        if baseline_stress > 0:
            stress_ratio = current_stress / baseline_stress
            stress_risk = min(100, (stress_ratio - 1.0) * 50 + stress_spikes * 5)
        
        manipulation_risk = min(100, manipulation_indicators * 15)
        timeline_risk = min(100, timeline_inconsistencies * 20)
        psychological_risk = min(100, psychological_red_flags * 12)

        # Calculate overall risk (weighted average)
        overall_risk = (
            contradiction_risk * 0.25 +
            stress_risk * 0.20 +
            manipulation_risk * 0.20 +
            timeline_risk * 0.20 +
            psychological_risk * 0.15
        )

        # Determine risk level
        if overall_risk >= 80:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk >= 60:
            risk_level = RiskLevel.HIGH
        elif overall_risk >= 40:
            risk_level = RiskLevel.MODERATE
        elif overall_risk >= 20:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL

        # Calculate confidence
        total_indicators = (contradiction_count + stress_spikes + 
                           manipulation_indicators + timeline_inconsistencies + 
                           psychological_red_flags)
        confidence = min(1.0, total_indicators / 10.0)

        # Determine trend
        risk_change = 0.0
        trend = "stable"
        if previous_assessment:
            risk_change = overall_risk - previous_assessment.overall_risk_score
            if risk_change > 10:
                trend = "increasing"
            elif risk_change < -10:
                trend = "decreasing"

        # Determine recommended action
        recommended_action = self._determine_escalation_action(
            risk_level, contradiction_count, manipulation_indicators,
            timeline_inconsistencies
        )

        # Generate rationale
        action_rationale = self._generate_action_rationale(
            recommended_action, contradiction_count, stress_spikes,
            manipulation_indicators, timeline_inconsistencies
        )

        risk = ParticipantRisk(
            participant_id=participant_id,
            participant_name=participant_name,
            contradiction_risk=contradiction_risk,
            stress_risk=stress_risk,
            manipulation_risk=manipulation_risk,
            timeline_risk=timeline_risk,
            psychological_risk=psychological_risk,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            confidence=confidence,
            risk_trend=trend,
            risk_change=risk_change,
            recommended_action=recommended_action,
            action_rationale=action_rationale
        )

        # Track history
        if participant_id not in self.participant_risks:
            self.participant_risks[participant_id] = []
        self.participant_risks[participant_id].append(risk)

        return risk

    def assess_case_risk(
        self,
        case_id: str,
        participant_a_risk: ParticipantRisk,
        participant_b_risk: ParticipantRisk,
        total_evidence: int,
        total_findings: int,
        days_under_investigation: int,
        previous_case_assessment: Optional[CaseRiskAssessment] = None
    ) -> CaseRiskAssessment:
        """
        Assess overall case risk

        Args:
            case_id: Case ID
            participant_a_risk: Risk for participant A
            participant_b_risk: Risk for participant B
            total_evidence: Total evidence items
            total_findings: Total forensic findings
            days_under_investigation: Days under investigation
            previous_case_assessment: Previous assessment for trend

        Returns:
            Case risk assessment
        """
        logger.info(f"Assessing case risk for {case_id}")

        # Combined risk (average of both participants, weighted by confidence)
        combined_risk = (
            (participant_a_risk.overall_risk_score * participant_a_risk.confidence +
             participant_b_risk.overall_risk_score * participant_b_risk.confidence) /
            (participant_a_risk.confidence + participant_b_risk.confidence)
        )

        # Determine case risk level
        if combined_risk >= 80:
            case_risk_level = RiskLevel.CRITICAL
        elif combined_risk >= 60:
            case_risk_level = RiskLevel.HIGH
        elif combined_risk >= 40:
            case_risk_level = RiskLevel.MODERATE
        elif combined_risk >= 20:
            case_risk_level = RiskLevel.LOW
        else:
            case_risk_level = RiskLevel.MINIMAL

        # Overall confidence
        overall_confidence = (participant_a_risk.confidence + participant_b_risk.confidence) / 2

        # Determine trend
        case_risk_change = 0.0
        case_trend = "stable"
        if previous_case_assessment:
            case_risk_change = combined_risk - previous_case_assessment.combined_risk_score
            if case_risk_change > 10:
                case_trend = "increasing"
            elif case_risk_change < -10:
                case_trend = "decreasing"

        # Identify critical factors
        critical_factors = self._identify_critical_factors(
            participant_a_risk, participant_b_risk
        )

        # Identify escalation triggers
        escalation_triggers = self._identify_escalation_triggers(
            participant_a_risk, participant_b_risk, total_findings
        )

        # Determine escalation action
        recommended_escalation = self._determine_case_escalation(
            case_risk_level, critical_factors, escalation_triggers
        )

        # Generate next steps
        next_steps = self._generate_next_steps(
            recommended_escalation, critical_factors, escalation_triggers
        )

        assessment = CaseRiskAssessment(
            case_id=case_id,
            assessment_timestamp=datetime.now().isoformat(),
            participant_a_risk=participant_a_risk,
            participant_b_risk=participant_b_risk,
            combined_risk_score=combined_risk,
            case_risk_level=case_risk_level,
            overall_confidence=overall_confidence,
            case_risk_trend=case_trend,
            case_risk_change=case_risk_change,
            critical_factors=critical_factors,
            escalation_triggers=escalation_triggers,
            recommended_escalation=recommended_escalation,
            recommended_next_steps=next_steps,
            days_under_investigation=days_under_investigation,
            evidence_count=total_evidence,
            findings_count=total_findings
        )

        self.assessments.append(assessment)
        return assessment

    def _determine_escalation_action(
        self,
        risk_level: RiskLevel,
        contradiction_count: int,
        manipulation_indicators: int,
        timeline_inconsistencies: int
    ) -> EscalationAction:
        """Determine recommended escalation action"""
        if risk_level == RiskLevel.CRITICAL:
            if contradiction_count > 5 and timeline_inconsistencies > 3:
                return EscalationAction.ARREST
            elif manipulation_indicators > 4:
                return EscalationAction.PROTECTIVE_ORDER
            else:
                return EscalationAction.INVESTIGATE

        elif risk_level == RiskLevel.HIGH:
            if contradiction_count > 3:
                return EscalationAction.CONFRONT
            else:
                return EscalationAction.INVESTIGATE

        elif risk_level == RiskLevel.MODERATE:
            return EscalationAction.INTERVIEW

        else:
            return EscalationAction.MONITOR

    def _generate_action_rationale(
        self,
        action: EscalationAction,
        contradiction_count: int,
        stress_spikes: int,
        manipulation_indicators: int,
        timeline_inconsistencies: int
    ) -> str:
        """Generate rationale for recommended action"""
        rationale = f"Recommended action: {action.value}. "

        if contradiction_count > 0:
            rationale += f"{contradiction_count} contradictions detected. "

        if stress_spikes > 0:
            rationale += f"{stress_spikes} stress spikes observed. "

        if manipulation_indicators > 0:
            rationale += f"{manipulation_indicators} manipulation indicators found. "

        if timeline_inconsistencies > 0:
            rationale += f"{timeline_inconsistencies} timeline inconsistencies identified. "

        return rationale

    def _identify_critical_factors(
        self,
        participant_a_risk: ParticipantRisk,
        participant_b_risk: ParticipantRisk
    ) -> List[str]:
        """Identify critical factors in case"""
        factors = []

        if participant_a_risk.risk_level == RiskLevel.CRITICAL:
            factors.append(f"{participant_a_risk.participant_name}: CRITICAL risk level")

        if participant_b_risk.risk_level == RiskLevel.CRITICAL:
            factors.append(f"{participant_b_risk.participant_name}: CRITICAL risk level")

        if participant_a_risk.contradiction_risk > 70:
            factors.append(f"{participant_a_risk.participant_name}: High contradiction risk")

        if participant_b_risk.contradiction_risk > 70:
            factors.append(f"{participant_b_risk.participant_name}: High contradiction risk")

        if participant_a_risk.manipulation_risk > 60:
            factors.append(f"{participant_a_risk.participant_name}: Manipulation tactics detected")

        if participant_b_risk.manipulation_risk > 60:
            factors.append(f"{participant_b_risk.participant_name}: Manipulation tactics detected")

        return factors

    def _identify_escalation_triggers(
        self,
        participant_a_risk: ParticipantRisk,
        participant_b_risk: ParticipantRisk,
        total_findings: int
    ) -> List[str]:
        """Identify escalation triggers"""
        triggers = []

        if total_findings > 10:
            triggers.append("Multiple forensic findings")

        if participant_a_risk.risk_trend == "increasing":
            triggers.append(f"{participant_a_risk.participant_name}: Risk increasing")

        if participant_b_risk.risk_trend == "increasing":
            triggers.append(f"{participant_b_risk.participant_name}: Risk increasing")

        if participant_a_risk.stress_risk > 70:
            triggers.append(f"{participant_a_risk.participant_name}: Elevated stress")

        if participant_b_risk.stress_risk > 70:
            triggers.append(f"{participant_b_risk.participant_name}: Elevated stress")

        return triggers

    def _determine_case_escalation(
        self,
        risk_level: RiskLevel,
        critical_factors: List[str],
        escalation_triggers: List[str]
    ) -> EscalationAction:
        """Determine case-level escalation action"""
        if risk_level == RiskLevel.CRITICAL:
            if len(critical_factors) > 2:
                return EscalationAction.ARREST
            else:
                return EscalationAction.INVESTIGATE

        elif risk_level == RiskLevel.HIGH:
            return EscalationAction.CONFRONT

        elif risk_level == RiskLevel.MODERATE:
            return EscalationAction.INTERVIEW

        else:
            return EscalationAction.MONITOR

    def _generate_next_steps(
        self,
        escalation: EscalationAction,
        critical_factors: List[str],
        escalation_triggers: List[str]
    ) -> List[str]:
        """Generate recommended next steps"""
        steps = []

        if escalation == EscalationAction.ARREST:
            steps.append("Prepare arrest warrant")
            steps.append("Coordinate with law enforcement")
            steps.append("Secure all evidence")

        elif escalation == EscalationAction.INVESTIGATE:
            steps.append("Conduct additional investigation")
            steps.append("Obtain missing evidence")
            steps.append("Interview additional witnesses")

        elif escalation == EscalationAction.CONFRONT:
            steps.append("Schedule confrontation interview")
            steps.append("Prepare evidence presentation")
            steps.append("Document responses")

        elif escalation == EscalationAction.INTERVIEW:
            steps.append("Schedule formal interview")
            steps.append("Prepare interview questions")
            steps.append("Record interview")

        elif escalation == EscalationAction.PROTECTIVE_ORDER:
            steps.append("Consult with legal team")
            steps.append("Prepare protective order petition")
            steps.append("Notify potential victim")

        else:  # MONITOR
            steps.append("Continue monitoring communications")
            steps.append("Schedule follow-up review")
            steps.append("Document any new evidence")

        return steps

    def get_current_assessment(self, case_id: str) -> Optional[CaseRiskAssessment]:
        """Get most recent assessment for case"""
        case_assessments = [a for a in self.assessments if a.case_id == case_id]
        return case_assessments[-1] if case_assessments else None

    def get_risk_trend(self, participant_id: str) -> List[float]:
        """Get risk score trend for participant"""
        if participant_id not in self.participant_risks:
            return []

        return [r.overall_risk_score for r in self.participant_risks[participant_id]]

    def export_assessment(self, assessment: CaseRiskAssessment, output_path: str) -> bool:
        """Export assessment to file"""
        try:
            import json

            data = {
                'case_id': assessment.case_id,
                'assessment_timestamp': assessment.assessment_timestamp,
                'combined_risk_score': assessment.combined_risk_score,
                'case_risk_level': assessment.case_risk_level.value,
                'overall_confidence': assessment.overall_confidence,
                'case_risk_trend': assessment.case_risk_trend,
                'case_risk_change': assessment.case_risk_change,
                'participant_a': {
                    'name': assessment.participant_a_risk.participant_name,
                    'risk_score': assessment.participant_a_risk.overall_risk_score,
                    'risk_level': assessment.participant_a_risk.risk_level.value,
                    'recommended_action': assessment.participant_a_risk.recommended_action.value
                },
                'participant_b': {
                    'name': assessment.participant_b_risk.participant_name,
                    'risk_score': assessment.participant_b_risk.overall_risk_score,
                    'risk_level': assessment.participant_b_risk.risk_level.value,
                    'recommended_action': assessment.participant_b_risk.recommended_action.value
                },
                'critical_factors': assessment.critical_factors,
                'escalation_triggers': assessment.escalation_triggers,
                'recommended_escalation': assessment.recommended_escalation.value,
                'recommended_next_steps': assessment.recommended_next_steps,
                'evidence_count': assessment.evidence_count,
                'findings_count': assessment.findings_count,
                'days_under_investigation': assessment.days_under_investigation
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported assessment to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    matrix = RiskEscalationMatrix()
    print("Risk Escalation Matrix initialized")
