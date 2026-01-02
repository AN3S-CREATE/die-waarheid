"""
Real-Time Alert System for Die Waarheid
Triggers alerts when high-risk findings are detected
Enables immediate investigative action

ALERT TYPES:
- Contradiction Alert: Statement conflicts detected
- Stress Alert: Stress levels spike significantly
- Timeline Alert: Timeline inconsistencies found
- Pattern Alert: Behavioral pattern changes detected
- Risk Alert: Overall case risk escalates
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    """Types of alerts"""
    CONTRADICTION = "contradiction"
    STRESS_SPIKE = "stress_spike"
    TIMELINE_INCONSISTENCY = "timeline_inconsistency"
    PATTERN_CHANGE = "pattern_change"
    RISK_ESCALATION = "risk_escalation"
    MANIPULATION_DETECTED = "manipulation_detected"
    AFRIKAANS_VERIFICATION_FAILED = "afrikaans_verification_failed"


@dataclass
class Alert:
    """Single alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: str
    
    # Content
    title: str
    description: str
    evidence_id: str
    participant_id: str
    
    # Details
    details: Dict[str, Any]
    confidence: float
    
    # Action
    recommended_action: str
    requires_immediate_attention: bool
    
    # Status
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp,
            'title': self.title,
            'description': self.description,
            'evidence_id': self.evidence_id,
            'participant_id': self.participant_id,
            'details': self.details,
            'confidence': self.confidence,
            'recommended_action': self.recommended_action,
            'requires_immediate_attention': self.requires_immediate_attention,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at,
            'acknowledged_by': self.acknowledged_by
        }


class AlertSystem:
    """
    Real-time alert system for investigation
    Monitors for high-risk findings and triggers alerts
    """

    def __init__(self):
        """Initialize alert system"""
        self.alert_counter = 0
        self.alerts: List[Alert] = []
        self.alert_queue: Queue = Queue()
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
        self.baseline_stress: Dict[str, float] = {}
        
        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default alert handlers"""
        for alert_type in AlertType:
            self.alert_handlers[alert_type] = []

    def register_handler(self, alert_type: AlertType, handler: Callable):
        """
        Register handler for alert type

        Args:
            alert_type: Type of alert
            handler: Callback function
        """
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)

    def check_contradiction(
        self,
        evidence_id: str,
        participant_id: str,
        current_statement: str,
        past_statement: str,
        confidence: float
    ) -> Optional[Alert]:
        """
        Check for contradictions and create alert

        Args:
            evidence_id: Current evidence ID
            participant_id: Participant ID
            current_statement: Current statement
            past_statement: Past statement
            confidence: Confidence of contradiction (0-1)

        Returns:
            Alert if created, None otherwise
        """
        if confidence < 0.75:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        # Determine severity
        if confidence > 0.95:
            severity = AlertSeverity.CRITICAL
        elif confidence > 0.85:
            severity = AlertSeverity.HIGH
        else:
            severity = AlertSeverity.MEDIUM

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.CONTRADICTION,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Statement Contradiction Detected",
            description=f"Participant contradicted previous statement with {confidence*100:.0f}% confidence",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'current_statement': current_statement[:200],
                'past_statement': past_statement[:200],
                'confidence': confidence
            },
            confidence=confidence,
            recommended_action="Review both statements and confront participant with discrepancy",
            requires_immediate_attention=confidence > 0.9
        )

        self._process_alert(alert)
        return alert

    def check_stress_spike(
        self,
        evidence_id: str,
        participant_id: str,
        current_stress: float,
        baseline_stress: float,
        spike_threshold: float = 1.5
    ) -> Optional[Alert]:
        """
        Check for stress level spikes

        Args:
            evidence_id: Evidence ID
            participant_id: Participant ID
            current_stress: Current stress level (0-100)
            baseline_stress: Baseline stress level
            spike_threshold: Multiplier for spike detection (default 1.5x)

        Returns:
            Alert if created, None otherwise
        """
        if baseline_stress == 0:
            return None

        spike_ratio = current_stress / baseline_stress
        
        if spike_ratio < spike_threshold:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        # Determine severity
        if spike_ratio > 3.0:
            severity = AlertSeverity.CRITICAL
        elif spike_ratio > 2.0:
            severity = AlertSeverity.HIGH
        else:
            severity = AlertSeverity.MEDIUM

        confidence = min(1.0, spike_ratio / 3.0)

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.STRESS_SPIKE,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Significant Stress Spike Detected",
            description=f"Stress level increased {spike_ratio:.1f}x above baseline",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'current_stress': current_stress,
                'baseline_stress': baseline_stress,
                'spike_ratio': spike_ratio
            },
            confidence=confidence,
            recommended_action="Investigate cause of stress increase - may indicate awareness of discrepancies",
            requires_immediate_attention=spike_ratio > 2.5
        )

        self._process_alert(alert)
        return alert

    def check_timeline_inconsistency(
        self,
        evidence_id: str,
        participant_id: str,
        claimed_time: str,
        conflicting_evidence: str,
        confidence: float
    ) -> Optional[Alert]:
        """
        Check for timeline inconsistencies

        Args:
            evidence_id: Evidence ID
            participant_id: Participant ID
            claimed_time: Claimed time of event
            conflicting_evidence: What conflicts with the claim
            confidence: Confidence of inconsistency (0-1)

        Returns:
            Alert if created, None otherwise
        """
        if confidence < 0.7:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        severity = AlertSeverity.CRITICAL if confidence > 0.9 else AlertSeverity.HIGH

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.TIMELINE_INCONSISTENCY,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Timeline Inconsistency Found",
            description=f"Claimed timeline conflicts with evidence ({confidence*100:.0f}% confidence)",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'claimed_time': claimed_time,
                'conflicting_evidence': conflicting_evidence,
                'confidence': confidence
            },
            confidence=confidence,
            recommended_action="Verify timeline with independent evidence and confront with discrepancy",
            requires_immediate_attention=True
        )

        self._process_alert(alert)
        return alert

    def check_pattern_change(
        self,
        evidence_id: str,
        participant_id: str,
        pattern_type: str,
        change_description: str,
        confidence: float
    ) -> Optional[Alert]:
        """
        Check for behavioral pattern changes

        Args:
            evidence_id: Evidence ID
            participant_id: Participant ID
            pattern_type: Type of pattern (vocabulary, tone, etc.)
            change_description: Description of change
            confidence: Confidence of change (0-1)

        Returns:
            Alert if created, None otherwise
        """
        if confidence < 0.65:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        severity = AlertSeverity.HIGH if confidence > 0.85 else AlertSeverity.MEDIUM

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.PATTERN_CHANGE,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Pattern Change Detected: {pattern_type}",
            description=f"{change_description} ({confidence*100:.0f}% confidence)",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'pattern_type': pattern_type,
                'change_description': change_description,
                'confidence': confidence
            },
            confidence=confidence,
            recommended_action="Note pattern change and monitor for escalation or deception indicators",
            requires_immediate_attention=confidence > 0.9
        )

        self._process_alert(alert)
        return alert

    def check_manipulation_detected(
        self,
        evidence_id: str,
        participant_id: str,
        manipulation_type: str,
        evidence_text: str,
        confidence: float
    ) -> Optional[Alert]:
        """
        Check for manipulation tactics

        Args:
            evidence_id: Evidence ID
            participant_id: Participant ID
            manipulation_type: Type of manipulation (gaslighting, guilt-tripping, etc.)
            evidence_text: Text showing manipulation
            confidence: Confidence (0-1)

        Returns:
            Alert if created, None otherwise
        """
        if confidence < 0.7:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        severity = AlertSeverity.CRITICAL if confidence > 0.9 else AlertSeverity.HIGH

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.MANIPULATION_DETECTED,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Manipulation Tactic Detected: {manipulation_type}",
            description=f"Evidence of {manipulation_type} detected ({confidence*100:.0f}% confidence)",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'manipulation_type': manipulation_type,
                'evidence_text': evidence_text[:300],
                'confidence': confidence
            },
            confidence=confidence,
            recommended_action=f"Document {manipulation_type} pattern and monitor for escalation",
            requires_immediate_attention=True
        )

        self._process_alert(alert)
        return alert

    def check_afrikaans_verification_failed(
        self,
        evidence_id: str,
        participant_id: str,
        confidence: float,
        reason: str
    ) -> Optional[Alert]:
        """
        Check for Afrikaans verification failures

        Args:
            evidence_id: Evidence ID
            participant_id: Participant ID
            confidence: Verification confidence (0-1)
            reason: Reason for low confidence

        Returns:
            Alert if created, None otherwise
        """
        if confidence > 0.7:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        severity = AlertSeverity.HIGH if confidence < 0.5 else AlertSeverity.MEDIUM

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.AFRIKAANS_VERIFICATION_FAILED,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Afrikaans Verification Failed",
            description=f"Afrikaans transcription confidence low: {confidence*100:.0f}% - {reason}",
            evidence_id=evidence_id,
            participant_id=participant_id,
            details={
                'confidence': confidence,
                'reason': reason
            },
            confidence=1.0 - confidence,
            recommended_action="Manual human review required for accurate transcription and translation",
            requires_immediate_attention=confidence < 0.5
        )

        self._process_alert(alert)
        return alert

    def check_risk_escalation(
        self,
        case_id: str,
        previous_risk: float,
        current_risk: float,
        escalation_threshold: float = 20.0
    ) -> Optional[Alert]:
        """
        Check for overall case risk escalation

        Args:
            case_id: Case ID
            previous_risk: Previous risk score
            current_risk: Current risk score
            escalation_threshold: Points increase to trigger alert

        Returns:
            Alert if created, None otherwise
        """
        risk_increase = current_risk - previous_risk
        
        if risk_increase < escalation_threshold:
            return None

        self.alert_counter += 1
        alert_id = f"ALT_{self.alert_counter:06d}"

        severity = AlertSeverity.CRITICAL if current_risk > 80 else AlertSeverity.HIGH

        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.RISK_ESCALATION,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            title=f"Case Risk Escalated",
            description=f"Overall case risk increased from {previous_risk:.0f} to {current_risk:.0f}",
            evidence_id=case_id,
            participant_id="",
            details={
                'previous_risk': previous_risk,
                'current_risk': current_risk,
                'increase': risk_increase
            },
            confidence=min(1.0, risk_increase / 50.0),
            recommended_action="Review all recent findings and consider escalating investigation level",
            requires_immediate_attention=current_risk > 75
        )

        self._process_alert(alert)
        return alert

    def _process_alert(self, alert: Alert):
        """Process alert and trigger handlers"""
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}")

        # Trigger handlers
        if alert.alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert.alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts"""
        return [a for a in self.alerts if not a.acknowledged]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]

    def get_alerts_for_participant(self, participant_id: str) -> List[Alert]:
        """Get alerts for specific participant"""
        return [a for a in self.alerts if a.participant_id == participant_id]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = ""):
        """
        Acknowledge alert

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged it
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now().isoformat()
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert acknowledged: {alert_id}")
                break

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        critical = len(self.get_critical_alerts())
        high = len([a for a in self.alerts if a.severity == AlertSeverity.HIGH])
        medium = len([a for a in self.alerts if a.severity == AlertSeverity.MEDIUM])
        low = len([a for a in self.alerts if a.severity == AlertSeverity.LOW])
        unacknowledged = len(self.get_unacknowledged_alerts())

        return {
            'total_alerts': len(self.alerts),
            'critical': critical,
            'high': high,
            'medium': medium,
            'low': low,
            'unacknowledged': unacknowledged,
            'requires_attention': critical > 0 or (high > 2)
        }

    def export_alerts(self, output_path) -> bool:
        """Export alerts to JSON"""
        try:
            data = {
                'exported_at': datetime.now().isoformat(),
                'summary': self.get_alert_summary(),
                'alerts': [a.to_dict() for a in self.alerts]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self.alerts)} alerts")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    system = AlertSystem()
    print("Alert System initialized")
