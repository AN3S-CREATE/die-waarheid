"""
Investigative Checklist Generation System for Die Waarheid
Auto-generates actionable next steps based on analysis findings
Turns forensic analysis into investigation strategy

CHECKLIST ITEMS:
- Specific questions to ask based on contradictions
- Evidence to verify or obtain
- Witnesses to interview
- Timeline gaps to fill
- Follow-up investigations needed
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ChecklistItemPriority(Enum):
    """Priority of checklist item"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChecklistItemType(Enum):
    """Type of checklist item"""
    QUESTION = "question"
    VERIFY_EVIDENCE = "verify_evidence"
    OBTAIN_EVIDENCE = "obtain_evidence"
    INTERVIEW = "interview"
    TIMELINE_GAP = "timeline_gap"
    FOLLOW_UP = "follow_up"
    CONFRONTATION = "confrontation"


@dataclass
class ChecklistItem:
    """Single checklist item"""
    item_id: str
    item_type: ChecklistItemType
    priority: ChecklistItemPriority
    
    # Content
    title: str
    description: str
    rationale: str
    
    # Target
    target_participant: str
    related_evidence: List[str]
    
    # Details
    specific_details: Dict[str, Any]
    expected_outcome: str
    
    # Status
    completed: bool = False
    completed_at: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'item_type': self.item_type.value,
            'priority': self.priority.value,
            'title': self.title,
            'description': self.description,
            'rationale': self.rationale,
            'target_participant': self.target_participant,
            'related_evidence': self.related_evidence,
            'specific_details': self.specific_details,
            'expected_outcome': self.expected_outcome,
            'completed': self.completed,
            'completed_at': self.completed_at,
            'notes': self.notes
        }


class InvestigativeChecklistGenerator:
    """
    Generates investigative checklists from analysis findings
    """

    def __init__(self):
        """Initialize checklist generator"""
        self.item_counter = 0
        self.checklists: Dict[str, List[ChecklistItem]] = {}

    def generate_checklist_from_findings(
        self,
        case_id: str,
        contradictions: List[Dict[str, Any]],
        pattern_changes: List[Dict[str, Any]],
        timeline_gaps: List[Dict[str, Any]],
        stress_spikes: List[Dict[str, Any]],
        manipulation_indicators: List[Dict[str, Any]],
        participants: List[Dict[str, str]]
    ) -> List[ChecklistItem]:
        """
        Generate comprehensive checklist from analysis findings

        Args:
            case_id: Case identifier
            contradictions: List of contradictions found
            pattern_changes: List of pattern changes
            timeline_gaps: List of timeline gaps
            stress_spikes: List of stress spikes
            manipulation_indicators: List of manipulation indicators
            participants: List of participants

        Returns:
            List of checklist items
        """
        logger.info(f"Generating checklist for case {case_id}")

        checklist = []

        # 1. Generate questions from contradictions
        for contradiction in contradictions:
            item = self._create_contradiction_question(
                contradiction,
                participants
            )
            if item:
                checklist.append(item)

        # 2. Generate timeline gap items
        for gap in timeline_gaps:
            item = self._create_timeline_gap_item(gap, participants)
            if item:
                checklist.append(item)

        # 3. Generate pattern change follow-ups
        for pattern in pattern_changes:
            item = self._create_pattern_follow_up(pattern, participants)
            if item:
                checklist.append(item)

        # 4. Generate stress spike investigations
        for spike in stress_spikes:
            item = self._create_stress_investigation(spike, participants)
            if item:
                checklist.append(item)

        # 5. Generate manipulation confrontations
        for manipulation in manipulation_indicators:
            item = self._create_manipulation_confrontation(manipulation, participants)
            if item:
                checklist.append(item)

        # 6. Generate evidence verification items
        checklist.extend(self._create_evidence_verification_items(
            contradictions, participants
        ))

        # Sort by priority
        checklist.sort(
            key=lambda x: (
                list(ChecklistItemPriority).index(x.priority),
                x.item_type.value
            )
        )

        self.checklists[case_id] = checklist
        logger.info(f"Generated {len(checklist)} checklist items")

        return checklist

    def _create_contradiction_question(
        self,
        contradiction: Dict[str, Any],
        participants: List[Dict[str, str]]
    ) -> Optional[ChecklistItem]:
        """Create question item from contradiction"""
        try:
            self.item_counter += 1
            item_id = f"CHK_{self.item_counter:06d}"

            participant_id = contradiction.get('participant_id', '')
            participant_name = self._get_participant_name(participant_id, participants)

            current = contradiction.get('current_statement', '')
            past = contradiction.get('past_statement', '')
            confidence = contradiction.get('confidence', 0.0)

            priority = ChecklistItemPriority.CRITICAL if confidence > 0.9 else ChecklistItemPriority.HIGH

            return ChecklistItem(
                item_id=item_id,
                item_type=ChecklistItemType.QUESTION,
                priority=priority,
                title=f"Confront {participant_name} with contradiction",
                description=f"Ask {participant_name} to explain contradiction between statements",
                rationale=f"Direct contradiction detected with {confidence*100:.0f}% confidence",
                target_participant=participant_id,
                related_evidence=contradiction.get('evidence_ids', []),
                specific_details={
                    'current_statement': current[:200],
                    'past_statement': past[:200],
                    'confidence': confidence,
                    'suggested_question': f"You previously said '{past[:100]}...' but now you're saying '{current[:100]}...'. Can you explain this difference?"
                },
                expected_outcome="Clarification of discrepancy or admission of inconsistency"
            )

        except Exception as e:
            logger.error(f"Error creating contradiction question: {e}")
            return None

    def _create_timeline_gap_item(
        self,
        gap: Dict[str, Any],
        participants: List[Dict[str, str]]
    ) -> Optional[ChecklistItem]:
        """Create timeline gap item"""
        try:
            self.item_counter += 1
            item_id = f"CHK_{self.item_counter:06d}"

            participant_id = gap.get('participant_id', '')
            participant_name = self._get_participant_name(participant_id, participants)
            gap_description = gap.get('description', '')
            gap_duration = gap.get('duration', 'unknown')

            return ChecklistItem(
                item_id=item_id,
                item_type=ChecklistItemType.TIMELINE_GAP,
                priority=ChecklistItemPriority.HIGH,
                title=f"Fill timeline gap for {participant_name}",
                description=f"Obtain evidence for gap: {gap_description}",
                rationale=f"Missing {gap_duration} in participant's timeline",
                target_participant=participant_id,
                related_evidence=gap.get('evidence_ids', []),
                specific_details={
                    'gap_description': gap_description,
                    'gap_duration': gap_duration,
                    'suggested_sources': [
                        'Phone records',
                        'Location data',
                        'Witness statements',
                        'Security footage',
                        'Transaction records'
                    ]
                },
                expected_outcome="Verification of participant's whereabouts during gap period"
            )

        except Exception as e:
            logger.error(f"Error creating timeline gap item: {e}")
            return None

    def _create_pattern_follow_up(
        self,
        pattern: Dict[str, Any],
        participants: List[Dict[str, str]]
    ) -> Optional[ChecklistItem]:
        """Create pattern change follow-up item"""
        try:
            self.item_counter += 1
            item_id = f"CHK_{self.item_counter:06d}"

            participant_id = pattern.get('participant_id', '')
            participant_name = self._get_participant_name(participant_id, participants)
            pattern_type = pattern.get('pattern_type', '')
            change = pattern.get('change_description', '')

            return ChecklistItem(
                item_id=item_id,
                item_type=ChecklistItemType.FOLLOW_UP,
                priority=ChecklistItemPriority.MEDIUM,
                title=f"Investigate {pattern_type} change in {participant_name}",
                description=f"Monitor and investigate pattern change: {change}",
                rationale=f"Significant {pattern_type} change detected",
                target_participant=participant_id,
                related_evidence=pattern.get('evidence_ids', []),
                specific_details={
                    'pattern_type': pattern_type,
                    'change_description': change,
                    'baseline': pattern.get('baseline', 'unknown'),
                    'current': pattern.get('current', 'unknown')
                },
                expected_outcome="Understanding of cause for pattern change"
            )

        except Exception as e:
            logger.error(f"Error creating pattern follow-up: {e}")
            return None

    def _create_stress_investigation(
        self,
        spike: Dict[str, Any],
        participants: List[Dict[str, str]]
    ) -> Optional[ChecklistItem]:
        """Create stress spike investigation item"""
        try:
            self.item_counter += 1
            item_id = f"CHK_{self.item_counter:06d}"

            participant_id = spike.get('participant_id', '')
            participant_name = self._get_participant_name(participant_id, participants)
            current_stress = spike.get('current_stress', 0)
            baseline = spike.get('baseline_stress', 0)
            ratio = current_stress / baseline if baseline > 0 else 0

            return ChecklistItem(
                item_id=item_id,
                item_type=ChecklistItemType.QUESTION,
                priority=ChecklistItemPriority.HIGH if ratio > 2.5 else ChecklistItemPriority.MEDIUM,
                title=f"Investigate stress spike in {participant_name}",
                description=f"Ask {participant_name} about cause of elevated stress",
                rationale=f"Stress level increased {ratio:.1f}x above baseline",
                target_participant=participant_id,
                related_evidence=spike.get('evidence_ids', []),
                specific_details={
                    'current_stress': current_stress,
                    'baseline_stress': baseline,
                    'spike_ratio': ratio,
                    'suggested_question': f"I noticed your stress level was significantly elevated in this message. What was causing that?"
                },
                expected_outcome="Explanation for stress increase or indication of deception"
            )

        except Exception as e:
            logger.error(f"Error creating stress investigation: {e}")
            return None

    def _create_manipulation_confrontation(
        self,
        manipulation: Dict[str, Any],
        participants: List[Dict[str, str]]
    ) -> Optional[ChecklistItem]:
        """Create manipulation confrontation item"""
        try:
            self.item_counter += 1
            item_id = f"CHK_{self.item_counter:06d}"

            participant_id = manipulation.get('participant_id', '')
            participant_name = self._get_participant_name(participant_id, participants)
            manip_type = manipulation.get('manipulation_type', '')
            evidence_text = manipulation.get('evidence_text', '')

            return ChecklistItem(
                item_id=item_id,
                item_type=ChecklistItemType.CONFRONTATION,
                priority=ChecklistItemPriority.CRITICAL,
                title=f"Confront {participant_name} about {manip_type}",
                description=f"Address {manip_type} behavior directly",
                rationale=f"Clear evidence of {manip_type} detected",
                target_participant=participant_id,
                related_evidence=manipulation.get('evidence_ids', []),
                specific_details={
                    'manipulation_type': manip_type,
                    'evidence_text': evidence_text[:300],
                    'suggested_approach': f"Point out the {manip_type} behavior and ask for explanation"
                },
                expected_outcome="Acknowledgment, denial, or escalation of behavior"
            )

        except Exception as e:
            logger.error(f"Error creating manipulation confrontation: {e}")
            return None

    def _create_evidence_verification_items(
        self,
        contradictions: List[Dict[str, Any]],
        participants: List[Dict[str, str]]
    ) -> List[ChecklistItem]:
        """Create evidence verification items"""
        items = []

        try:
            # Get unique evidence IDs from contradictions
            evidence_ids = set()
            for contradiction in contradictions:
                evidence_ids.update(contradiction.get('evidence_ids', []))

            for evidence_id in evidence_ids:
                self.item_counter += 1
                item_id = f"CHK_{self.item_counter:06d}"

                item = ChecklistItem(
                    item_id=item_id,
                    item_type=ChecklistItemType.VERIFY_EVIDENCE,
                    priority=ChecklistItemPriority.HIGH,
                    title=f"Verify evidence {evidence_id}",
                    description=f"Independently verify authenticity and accuracy of {evidence_id}",
                    rationale="Evidence involved in contradictions requires verification",
                    target_participant="",
                    related_evidence=[evidence_id],
                    specific_details={
                        'evidence_id': evidence_id,
                        'verification_methods': [
                            'Check source authenticity',
                            'Verify timestamps',
                            'Confirm content accuracy',
                            'Check for tampering'
                        ]
                    },
                    expected_outcome="Confirmation of evidence authenticity and reliability"
                )
                items.append(item)

        except Exception as e:
            logger.error(f"Error creating evidence verification items: {e}")

        return items

    def _get_participant_name(
        self,
        participant_id: str,
        participants: List[Dict[str, str]]
    ) -> str:
        """Get participant name from ID"""
        for p in participants:
            if p.get('participant_id') == participant_id:
                return p.get('primary_username', participant_id)
        return participant_id

    def mark_item_complete(
        self,
        case_id: str,
        item_id: str,
        notes: str = ""
    ):
        """Mark checklist item as complete"""
        if case_id in self.checklists:
            for item in self.checklists[case_id]:
                if item.item_id == item_id:
                    item.completed = True
                    item.completed_at = datetime.now().isoformat()
                    item.notes = notes
                    logger.info(f"Marked item complete: {item_id}")
                    break

    def get_pending_items(self, case_id: str) -> List[ChecklistItem]:
        """Get all pending checklist items"""
        if case_id not in self.checklists:
            return []
        return [item for item in self.checklists[case_id] if not item.completed]

    def get_critical_items(self, case_id: str) -> List[ChecklistItem]:
        """Get all critical priority items"""
        if case_id not in self.checklists:
            return []
        return [
            item for item in self.checklists[case_id]
            if item.priority == ChecklistItemPriority.CRITICAL
        ]

    def get_items_for_participant(
        self,
        case_id: str,
        participant_id: str
    ) -> List[ChecklistItem]:
        """Get checklist items for specific participant"""
        if case_id not in self.checklists:
            return []
        return [
            item for item in self.checklists[case_id]
            if item.target_participant == participant_id
        ]

    def get_checklist_summary(self, case_id: str) -> Dict[str, Any]:
        """Get checklist summary"""
        if case_id not in self.checklists:
            return {'total_items': 0}

        checklist = self.checklists[case_id]
        completed = len([i for i in checklist if i.completed])
        pending = len([i for i in checklist if not i.completed])
        critical = len([i for i in checklist if i.priority == ChecklistItemPriority.CRITICAL])

        return {
            'total_items': len(checklist),
            'completed': completed,
            'pending': pending,
            'critical_pending': len([i for i in checklist if not i.completed and i.priority == ChecklistItemPriority.CRITICAL]),
            'completion_percentage': (completed / len(checklist) * 100) if checklist else 0,
            'items_by_type': self._count_by_type(checklist),
            'items_by_priority': self._count_by_priority(checklist)
        }

    def _count_by_type(self, checklist: List[ChecklistItem]) -> Dict[str, int]:
        """Count items by type"""
        counts = {}
        for item in checklist:
            item_type = item.item_type.value
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts

    def _count_by_priority(self, checklist: List[ChecklistItem]) -> Dict[str, int]:
        """Count items by priority"""
        counts = {}
        for item in checklist:
            priority = item.priority.value
            counts[priority] = counts.get(priority, 0) + 1
        return counts

    def export_checklist(self, case_id: str, output_path) -> bool:
        """Export checklist to file"""
        try:
            if case_id not in self.checklists:
                return False

            checklist = self.checklists[case_id]
            data = {
                'case_id': case_id,
                'exported_at': datetime.now().isoformat(),
                'summary': self.get_checklist_summary(case_id),
                'items': [item.to_dict() for item in checklist]
            }

            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported checklist to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    generator = InvestigativeChecklistGenerator()
    print("Investigative Checklist Generator initialized")
