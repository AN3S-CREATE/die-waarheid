"""
Narrative Reconstruction System for Die Waarheid
Reconstructs each participant's version of events
Identifies gaps, inconsistencies, and alternative explanations

FEATURES:
- Automatic narrative extraction from statements
- Timeline of claimed events
- Gap identification
- Inconsistency detection
- Alternative narrative generation
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NarrativeElement(Enum):
    """Type of narrative element"""
    EVENT = "event"
    CLAIM = "claim"
    EMOTION = "emotion"
    JUSTIFICATION = "justification"
    DENIAL = "denial"


@dataclass
class NarrativeEvent:
    """Single event in participant's narrative"""
    event_id: str
    participant_id: str
    claimed_time: datetime
    event_description: str
    location: Optional[str]
    people_involved: List[str]
    evidence_supporting: List[str]
    evidence_contradicting: List[str]
    confidence: float


@dataclass
class ParticipantNarrative:
    """Complete narrative for one participant"""
    participant_id: str
    participant_name: str
    
    # Narrative elements
    events: List[NarrativeEvent]
    key_claims: List[str]
    justifications: List[str]
    denials: List[str]
    
    # Analysis
    narrative_consistency: float
    timeline_consistency: float
    gaps: List[Dict[str, Any]]
    inconsistencies: List[Dict[str, Any]]
    
    # Summary
    overall_narrative: str
    alternative_narratives: List[str]


class NarrativeReconstructor:
    """
    Reconstructs participant narratives from statements
    """

    def __init__(self):
        """Initialize reconstructor"""
        self.narratives: Dict[str, ParticipantNarrative] = {}
        self.event_counter = 0

    def build_narrative(
        self,
        participant_id: str,
        participant_name: str,
        statements: List[Dict[str, Any]],
        timeline_data: Optional[List[Dict[str, Any]]] = None
    ) -> ParticipantNarrative:
        """
        Build narrative from participant statements

        Args:
            participant_id: Participant ID
            participant_name: Participant name
            statements: List of statements with content, timestamp, etc.
            timeline_data: Optional timeline data for consistency checking

        Returns:
            Reconstructed narrative
        """
        logger.info(f"Building narrative for {participant_name}")

        events = []
        key_claims = []
        justifications = []
        denials = []

        # Extract events and claims from statements
        for stmt in statements:
            content = stmt.get('content', '')
            timestamp = stmt.get('timestamp')
            evidence_id = stmt.get('evidence_id', '')

            # Extract events
            extracted_events = self._extract_events(content, timestamp, participant_id)
            events.extend(extracted_events)

            # Extract claims
            claims = self._extract_claims(content)
            key_claims.extend(claims)

            # Extract justifications
            justs = self._extract_justifications(content)
            justifications.extend(justs)

            # Extract denials
            denies = self._extract_denials(content)
            denials.extend(denies)

        # Identify gaps
        gaps = self._identify_gaps(events, timeline_data)

        # Identify inconsistencies
        inconsistencies = self._identify_inconsistencies(events, key_claims)

        # Calculate consistency scores
        narrative_consistency = self._calculate_narrative_consistency(events, key_claims)
        timeline_consistency = self._calculate_timeline_consistency(events)

        # Generate overall narrative
        overall_narrative = self._generate_overall_narrative(
            participant_name, events, key_claims, justifications
        )

        # Generate alternative narratives
        alternative_narratives = self._generate_alternative_narratives(
            participant_name, events, inconsistencies, gaps
        )

        narrative = ParticipantNarrative(
            participant_id=participant_id,
            participant_name=participant_name,
            events=events,
            key_claims=key_claims,
            justifications=justifications,
            denials=denials,
            narrative_consistency=narrative_consistency,
            timeline_consistency=timeline_consistency,
            gaps=gaps,
            inconsistencies=inconsistencies,
            overall_narrative=overall_narrative,
            alternative_narratives=alternative_narratives
        )

        self.narratives[participant_id] = narrative
        return narrative

    def _extract_events(
        self,
        content: str,
        timestamp: datetime,
        participant_id: str
    ) -> List[NarrativeEvent]:
        """Extract events from statement"""
        events = []

        # Simple event extraction (can be enhanced with NLP)
        event_keywords = [
            'was', 'went', 'did', 'happened', 'occurred', 'saw', 'met',
            'talked', 'called', 'texted', 'arrived', 'left', 'stayed'
        ]

        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains event keywords
            if any(keyword in sentence.lower() for keyword in event_keywords):
                self.event_counter += 1
                event = NarrativeEvent(
                    event_id=f"EVT_{self.event_counter:06d}",
                    participant_id=participant_id,
                    claimed_time=timestamp,
                    event_description=sentence,
                    location=self._extract_location(sentence),
                    people_involved=self._extract_people(sentence),
                    evidence_supporting=[],
                    evidence_contradicting=[],
                    confidence=0.6
                )
                events.append(event)

        return events

    def _extract_claims(self, content: str) -> List[str]:
        """Extract key claims from statement"""
        claims = []

        # Simple claim extraction
        claim_patterns = [
            'I was', 'I am', 'I did', 'I never', 'I always',
            'I remember', 'I know', 'I believe', 'I think'
        ]

        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(pattern in sentence for pattern in claim_patterns):
                claims.append(sentence)

        return claims

    def _extract_justifications(self, content: str) -> List[str]:
        """Extract justifications from statement"""
        justifications = []

        justification_keywords = [
            'because', 'since', 'reason', 'why', 'due to', 'caused by'
        ]

        sentences = content.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in justification_keywords):
                justifications.append(sentence.strip())

        return justifications

    def _extract_denials(self, content: str) -> List[str]:
        """Extract denials from statement"""
        denials = []

        denial_patterns = [
            'I never', 'I didn\'t', 'I don\'t', 'I wouldn\'t',
            'not true', 'false', 'wrong', 'that\'s not'
        ]

        sentences = content.split('.')
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in denial_patterns):
                denials.append(sentence.strip())

        return denials

    def _extract_location(self, sentence: str) -> Optional[str]:
        """Extract location from sentence"""
        location_keywords = [
            'at', 'in', 'near', 'by', 'around', 'outside', 'inside',
            'home', 'office', 'work', 'place', 'house', 'street'
        ]

        for keyword in location_keywords:
            if keyword in sentence.lower():
                # Simple extraction - could be enhanced
                parts = sentence.lower().split(keyword)
                if len(parts) > 1:
                    return parts[1].split('.')[0].strip()

        return None

    def _extract_people(self, sentence: str) -> List[str]:
        """Extract people mentioned in sentence"""
        people = []

        # Simple extraction - could be enhanced with NER
        person_keywords = ['he', 'she', 'they', 'person', 'guy', 'girl', 'man', 'woman']

        for keyword in person_keywords:
            if keyword in sentence.lower():
                people.append(keyword)

        return list(set(people))

    def _identify_gaps(
        self,
        events: List[NarrativeEvent],
        timeline_data: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify gaps in narrative"""
        gaps = []

        if not timeline_data:
            return gaps

        # Check for time periods without events
        timeline_dates = [datetime.fromisoformat(t['timestamp']) for t in timeline_data]
        event_dates = [e.claimed_time for e in events]

        if timeline_dates:
            min_date = min(timeline_dates)
            max_date = max(timeline_dates)

            # Look for gaps > 24 hours
            sorted_events = sorted(event_dates)
            for i in range(len(sorted_events) - 1):
                gap = (sorted_events[i + 1] - sorted_events[i]).total_seconds() / 3600
                if gap > 24:
                    gaps.append({
                        'start': sorted_events[i].isoformat(),
                        'end': sorted_events[i + 1].isoformat(),
                        'duration_hours': gap,
                        'description': f"No events reported for {gap:.0f} hours"
                    })

        return gaps

    def _identify_inconsistencies(
        self,
        events: List[NarrativeEvent],
        claims: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify inconsistencies in narrative"""
        inconsistencies = []

        # Check for contradictory claims
        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i + 1:], start=i + 1):
                # Simple contradiction detection
                if 'never' in claim1.lower() and 'always' in claim2.lower():
                    inconsistencies.append({
                        'claim_a': claim1,
                        'claim_b': claim2,
                        'type': 'direct_contradiction',
                        'severity': 'high'
                    })
                elif 'didn\'t' in claim1.lower() and 'did' in claim2.lower():
                    inconsistencies.append({
                        'claim_a': claim1,
                        'claim_b': claim2,
                        'type': 'action_contradiction',
                        'severity': 'high'
                    })

        return inconsistencies

    def _calculate_narrative_consistency(
        self,
        events: List[NarrativeEvent],
        claims: List[str]
    ) -> float:
        """Calculate narrative consistency score (0-1)"""
        if not events or not claims:
            return 0.5

        # Score based on number of inconsistencies
        consistency = 1.0
        
        # Penalize for contradictions
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i + 1:]:
                if ('never' in claim1.lower() and 'always' in claim2.lower()) or \
                   ('didn\'t' in claim1.lower() and 'did' in claim2.lower()):
                    consistency -= 0.1

        return max(0.0, consistency)

    def _calculate_timeline_consistency(self, events: List[NarrativeEvent]) -> float:
        """Calculate timeline consistency score (0-1)"""
        if not events:
            return 0.5

        # Check if events are in chronological order
        sorted_events = sorted(events, key=lambda x: x.claimed_time)
        
        in_order = sum(1 for i, event in enumerate(events) if event == sorted_events[i])
        consistency = in_order / len(events)

        return consistency

    def _generate_overall_narrative(
        self,
        participant_name: str,
        events: List[NarrativeEvent],
        claims: List[str],
        justifications: List[str]
    ) -> str:
        """Generate overall narrative summary"""
        narrative = f"{participant_name}'s account:\n\n"

        if events:
            narrative += "Key events claimed:\n"
            for event in sorted(events, key=lambda x: x.claimed_time)[:5]:
                narrative += f"- {event.event_description}\n"

        if claims:
            narrative += "\nMain claims:\n"
            for claim in claims[:3]:
                narrative += f"- {claim}\n"

        if justifications:
            narrative += "\nJustifications provided:\n"
            for just in justifications[:2]:
                narrative += f"- {just}\n"

        return narrative

    def _generate_alternative_narratives(
        self,
        participant_name: str,
        events: List[NarrativeEvent],
        inconsistencies: List[Dict[str, Any]],
        gaps: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate alternative narrative explanations"""
        alternatives = []

        if inconsistencies:
            alt = f"Alternative: {participant_name} is being inconsistent about key facts. "
            alt += f"The contradictions suggest either memory issues or deliberate deception."
            alternatives.append(alt)

        if gaps:
            alt = f"Alternative: {participant_name} has significant gaps in their account. "
            alt += f"These gaps could be filled with activities they're not disclosing."
            alternatives.append(alt)

        if len(events) < 3:
            alt = f"Alternative: {participant_name}'s account is sparse on details. "
            alt += f"More specific information is needed to verify the narrative."
            alternatives.append(alt)

        return alternatives

    def get_narrative(self, participant_id: str) -> Optional[ParticipantNarrative]:
        """Get narrative for participant"""
        return self.narratives.get(participant_id)

    def compare_narratives(
        self,
        participant_a_id: str,
        participant_b_id: str
    ) -> Dict[str, Any]:
        """
        Compare two participant narratives

        Args:
            participant_a_id: First participant ID
            participant_b_id: Second participant ID

        Returns:
            Comparison analysis
        """
        narrative_a = self.narratives.get(participant_a_id)
        narrative_b = self.narratives.get(participant_b_id)

        if not narrative_a or not narrative_b:
            return {}

        # Find overlapping events
        overlapping_events = []
        for event_a in narrative_a.events:
            for event_b in narrative_b.events:
                # Check if events are about same time period
                time_diff = abs((event_a.claimed_time - event_b.claimed_time).total_seconds() / 3600)
                if time_diff < 24:  # Within 24 hours
                    overlapping_events.append({
                        'participant_a_event': event_a.event_description,
                        'participant_b_event': event_b.event_description,
                        'time_difference_hours': time_diff,
                        'agreement': self._check_event_agreement(event_a, event_b)
                    })

        return {
            'participant_a': narrative_a.participant_name,
            'participant_b': narrative_b.participant_name,
            'overlapping_events': overlapping_events,
            'narrative_a_consistency': narrative_a.narrative_consistency,
            'narrative_b_consistency': narrative_b.narrative_consistency,
            'timeline_a_consistency': narrative_a.timeline_consistency,
            'timeline_b_consistency': narrative_b.timeline_consistency,
            'agreement_level': self._calculate_agreement_level(overlapping_events)
        }

    def _check_event_agreement(
        self,
        event_a: NarrativeEvent,
        event_b: NarrativeEvent
    ) -> str:
        """Check if two events agree"""
        desc_a = event_a.event_description.lower()
        desc_b = event_b.event_description.lower()

        # Simple agreement check
        common_words = len(set(desc_a.split()) & set(desc_b.split()))
        if common_words > 3:
            return "high_agreement"
        elif common_words > 1:
            return "partial_agreement"
        else:
            return "disagreement"

    def _calculate_agreement_level(self, overlapping_events: List[Dict]) -> float:
        """Calculate overall agreement level"""
        if not overlapping_events:
            return 0.0

        agreement_scores = {
            'high_agreement': 1.0,
            'partial_agreement': 0.5,
            'disagreement': 0.0
        }

        total = sum(agreement_scores.get(e['agreement'], 0) for e in overlapping_events)
        return total / len(overlapping_events)

    def export_narrative(self, participant_id: str, output_path: str) -> bool:
        """Export narrative to file"""
        try:
            narrative = self.narratives.get(participant_id)
            if not narrative:
                return False

            import json
            data = {
                'participant_id': narrative.participant_id,
                'participant_name': narrative.participant_name,
                'overall_narrative': narrative.overall_narrative,
                'narrative_consistency': narrative.narrative_consistency,
                'timeline_consistency': narrative.timeline_consistency,
                'gaps': narrative.gaps,
                'inconsistencies': narrative.inconsistencies,
                'alternative_narratives': narrative.alternative_narratives,
                'key_claims': narrative.key_claims,
                'denials': narrative.denials
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported narrative to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    reconstructor = NarrativeReconstructor()
    print("Narrative Reconstructor initialized")
