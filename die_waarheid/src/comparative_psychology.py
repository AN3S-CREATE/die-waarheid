"""
Comparative Psychology Profiles for Die Waarheid
Builds side-by-side psychological profiles of participants
Shows behavioral differences, manipulation tactics, emotional patterns

FEATURES:
- Individual psychological profiles
- Side-by-side comparison
- Stress response patterns
- Manipulation tactics unique to each participant
- Emotional escalation patterns
- Consistency tracking
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BehaviorPattern(Enum):
    """Type of behavior pattern"""
    STRESS_RESPONSE = "stress_response"
    MANIPULATION = "manipulation"
    EMOTIONAL_ESCALATION = "emotional_escalation"
    DEFENSIVENESS = "defensiveness"
    VICTIM_MENTALITY = "victim_mentality"
    ACCOUNTABILITY = "accountability"
    CONSISTENCY = "consistency"


@dataclass
class PsychologicalProfile:
    """Psychological profile for one participant"""
    participant_id: str
    participant_name: str
    
    # Stress patterns
    baseline_stress: float
    stress_spike_frequency: int
    stress_spike_magnitude: float
    stress_triggers: List[str]
    
    # Manipulation tactics
    gaslighting_indicators: List[str]
    guilt_tripping_indicators: List[str]
    threatening_language: List[str]
    isolation_tactics: List[str]
    love_bombing_indicators: List[str]
    
    # Emotional patterns
    emotional_tone_baseline: str
    emotional_escalation_rate: float
    emotional_consistency: float
    anger_triggers: List[str]
    fear_indicators: List[str]
    
    # Behavioral traits
    defensiveness_score: float
    accountability_score: float
    victim_mentality_score: float
    consistency_score: float
    
    # Summary
    psychological_profile_summary: str
    risk_indicators: List[str]
    behavioral_red_flags: List[str]


@dataclass
class ComparativeAnalysis:
    """Comparison between two participants"""
    participant_a_id: str
    participant_b_id: str
    participant_a_name: str
    participant_b_name: str
    
    # Stress comparison
    stress_baseline_difference: float
    stress_response_pattern_difference: str
    
    # Manipulation comparison
    manipulation_tactics_a: List[str]
    manipulation_tactics_b: List[str]
    unique_tactics_a: List[str]
    unique_tactics_b: List[str]
    
    # Emotional comparison
    emotional_stability_a: float
    emotional_stability_b: float
    emotional_pattern_difference: str
    
    # Behavioral comparison
    defensiveness_difference: float
    accountability_difference: float
    victim_mentality_difference: float
    consistency_difference: float
    
    # Summary
    key_differences: List[str]
    behavioral_contrast: str
    risk_assessment: Dict[str, str]


class ComparativePsychologyAnalyzer:
    """
    Analyzes and compares psychological profiles
    """

    def __init__(self):
        """Initialize analyzer"""
        self.profiles: Dict[str, PsychologicalProfile] = {}
        self.comparisons: Dict[str, ComparativeAnalysis] = {}

    def build_profile(
        self,
        participant_id: str,
        participant_name: str,
        evidence_data: List[Dict[str, Any]]
    ) -> PsychologicalProfile:
        """
        Build psychological profile from evidence

        Args:
            participant_id: Participant ID
            participant_name: Participant name
            evidence_data: List of evidence with stress, text, etc.

        Returns:
            Psychological profile
        """
        logger.info(f"Building psychological profile for {participant_name}")

        # Extract stress patterns
        stress_levels = [e.get('stress_level', 0) for e in evidence_data if e.get('stress_level')]
        baseline_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 0
        stress_spikes = len([s for s in stress_levels if s > baseline_stress * 1.5])
        avg_spike_magnitude = sum(s / baseline_stress for s in stress_levels if s > baseline_stress * 1.5) / max(stress_spikes, 1) if stress_spikes > 0 else 0

        # Extract manipulation indicators
        gaslighting = self._detect_gaslighting(evidence_data)
        guilt_tripping = self._detect_guilt_tripping(evidence_data)
        threatening = self._detect_threatening_language(evidence_data)
        isolation = self._detect_isolation_tactics(evidence_data)
        love_bombing = self._detect_love_bombing(evidence_data)

        # Extract emotional patterns
        emotional_tone = self._analyze_emotional_tone(evidence_data)
        escalation_rate = self._calculate_escalation_rate(evidence_data)
        emotional_consistency = self._calculate_emotional_consistency(evidence_data)
        anger_triggers = self._identify_anger_triggers(evidence_data)
        fear_indicators = self._identify_fear_indicators(evidence_data)

        # Calculate behavioral scores
        defensiveness = self._calculate_defensiveness(evidence_data)
        accountability = self._calculate_accountability(evidence_data)
        victim_mentality = self._calculate_victim_mentality(evidence_data)
        consistency = self._calculate_consistency(evidence_data)

        # Generate summary
        profile_summary = self._generate_profile_summary(
            participant_name, baseline_stress, gaslighting, guilt_tripping,
            defensiveness, accountability, victim_mentality
        )

        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(
            gaslighting, guilt_tripping, threatening, isolation,
            defensiveness, victim_mentality
        )

        # Identify red flags
        red_flags = self._identify_red_flags(
            stress_spikes, escalation_rate, consistency, accountability
        )

        profile = PsychologicalProfile(
            participant_id=participant_id,
            participant_name=participant_name,
            baseline_stress=baseline_stress,
            stress_spike_frequency=stress_spikes,
            stress_spike_magnitude=avg_spike_magnitude,
            stress_triggers=self._identify_stress_triggers(evidence_data),
            gaslighting_indicators=gaslighting,
            guilt_tripping_indicators=guilt_tripping,
            threatening_language=threatening,
            isolation_tactics=isolation,
            love_bombing_indicators=love_bombing,
            emotional_tone_baseline=emotional_tone,
            emotional_escalation_rate=escalation_rate,
            emotional_consistency=emotional_consistency,
            anger_triggers=anger_triggers,
            fear_indicators=fear_indicators,
            defensiveness_score=defensiveness,
            accountability_score=accountability,
            victim_mentality_score=victim_mentality,
            consistency_score=consistency,
            psychological_profile_summary=profile_summary,
            risk_indicators=risk_indicators,
            behavioral_red_flags=red_flags
        )

        self.profiles[participant_id] = profile
        return profile

    def _detect_gaslighting(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Detect gaslighting language patterns"""
        gaslighting_phrases = [
            "you're imagining things", "that never happened", "you're too sensitive",
            "you're crazy", "you're making it up", "that's not what I said",
            "you're overreacting", "you're being dramatic", "you misunderstood"
        ]

        indicators = []
        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for phrase in gaslighting_phrases:
                if phrase in text:
                    indicators.append(phrase)

        return list(set(indicators))

    def _detect_guilt_tripping(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Detect guilt-tripping language"""
        guilt_phrases = [
            "after all i've done", "i sacrificed everything", "you don't appreciate",
            "you're ungrateful", "i gave up", "you owe me", "i did this for you"
        ]

        indicators = []
        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for phrase in guilt_phrases:
                if phrase in text:
                    indicators.append(phrase)

        return list(set(indicators))

    def _detect_threatening_language(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Detect threatening language"""
        threat_phrases = [
            "you'll regret", "i'll make sure", "you'll see", "just wait",
            "i'll tell everyone", "i'll destroy you", "you'll lose everything"
        ]

        indicators = []
        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for phrase in threat_phrases:
                if phrase in text:
                    indicators.append(phrase)

        return list(set(indicators))

    def _detect_isolation_tactics(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Detect isolation tactics"""
        isolation_phrases = [
            "don't talk to them", "they're against you", "nobody understands",
            "only i can help", "don't trust anyone", "they're lying about me"
        ]

        indicators = []
        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for phrase in isolation_phrases:
                if phrase in text:
                    indicators.append(phrase)

        return list(set(indicators))

    def _detect_love_bombing(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Detect love-bombing language"""
        love_bomb_phrases = [
            "i love you so much", "you're perfect", "you're the only one",
            "i can't live without you", "you complete me", "you're my everything"
        ]

        indicators = []
        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for phrase in love_bomb_phrases:
                if phrase in text:
                    indicators.append(phrase)

        return list(set(indicators))

    def _analyze_emotional_tone(self, evidence_data: List[Dict[str, Any]]) -> str:
        """Analyze baseline emotional tone"""
        emotional_words = {
            'angry': ['angry', 'furious', 'rage', 'hate'],
            'sad': ['sad', 'depressed', 'miserable', 'heartbroken'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared'],
            'defensive': ['defensive', 'accused', 'blamed', 'unfair'],
            'neutral': []
        }

        tone_counts = {tone: 0 for tone in emotional_words.keys()}

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            for tone, words in emotional_words.items():
                for word in words:
                    if word in text:
                        tone_counts[tone] += 1

        # Return most common tone
        max_tone = max(tone_counts, key=tone_counts.get)
        return max_tone if tone_counts[max_tone] > 0 else 'neutral'

    def _calculate_escalation_rate(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate emotional escalation rate"""
        stress_levels = [e.get('stress_level', 0) for e in evidence_data if e.get('stress_level')]
        
        if len(stress_levels) < 2:
            return 0.0

        # Calculate average increase per message
        increases = []
        for i in range(len(stress_levels) - 1):
            increase = stress_levels[i + 1] - stress_levels[i]
            if increase > 0:
                increases.append(increase)

        return sum(increases) / len(increases) if increases else 0.0

    def _calculate_emotional_consistency(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate emotional consistency (0-1)"""
        stress_levels = [e.get('stress_level', 0) for e in evidence_data if e.get('stress_level')]
        
        if len(stress_levels) < 2:
            return 0.5

        # Calculate standard deviation
        avg = sum(stress_levels) / len(stress_levels)
        variance = sum((x - avg) ** 2 for x in stress_levels) / len(stress_levels)
        std_dev = variance ** 0.5

        # Normalize: lower std dev = higher consistency
        consistency = 1.0 / (1.0 + std_dev / 100.0)
        return consistency

    def _identify_stress_triggers(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Identify what triggers stress"""
        triggers = []

        for evidence in evidence_data:
            if evidence.get('stress_level', 0) > 60:
                text = evidence.get('text', '')
                # Extract potential triggers (simple heuristic)
                if 'you' in text.lower():
                    triggers.append("confrontation")
                if 'why' in text.lower() or 'how' in text.lower():
                    triggers.append("questioning")
                if 'remember' in text.lower():
                    triggers.append("memory recall")

        return list(set(triggers))

    def _identify_anger_triggers(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Identify anger triggers"""
        triggers = []
        anger_words = ['angry', 'furious', 'rage', 'hate', 'mad']

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            if any(word in text for word in anger_words):
                triggers.append(evidence.get('context', 'unknown'))

        return list(set(triggers))[:3]

    def _identify_fear_indicators(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Identify fear indicators"""
        indicators = []
        fear_words = ['scared', 'afraid', 'terrified', 'worried', 'anxious']

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            if any(word in text for word in fear_words):
                indicators.append(text[:100])

        return indicators[:3]

    def _calculate_defensiveness(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate defensiveness score (0-1)"""
        defensive_words = ['but', 'however', 'actually', 'no', 'never', 'didn\'t']
        defensive_count = 0
        total_words = 0

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            words = text.split()
            total_words += len(words)
            defensive_count += sum(1 for word in words if word in defensive_words)

        return defensive_count / max(total_words, 1) if total_words > 0 else 0.0

    def _calculate_accountability(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate accountability score (0-1)"""
        accountability_words = ['i was wrong', 'my fault', 'i apologize', 'i\'m sorry', 'i made a mistake']
        accountability_count = 0

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            accountability_count += sum(1 for phrase in accountability_words if phrase in text)

        return min(1.0, accountability_count / max(len(evidence_data), 1))

    def _calculate_victim_mentality(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate victim mentality score (0-1)"""
        victim_words = ['always blame me', 'never my fault', 'everyone against me', 'unfair', 'victim']
        victim_count = 0

        for evidence in evidence_data:
            text = evidence.get('text', '').lower()
            victim_count += sum(1 for word in victim_words if word in text)

        return min(1.0, victim_count / max(len(evidence_data), 1))

    def _calculate_consistency(self, evidence_data: List[Dict[str, Any]]) -> float:
        """Calculate consistency score (0-1)"""
        # Based on contradiction count
        contradictions = sum(1 for e in evidence_data if e.get('has_contradiction', False))
        return 1.0 - (contradictions / max(len(evidence_data), 1))

    def _generate_profile_summary(
        self,
        name: str,
        stress: float,
        gaslighting: List[str],
        guilt_tripping: List[str],
        defensiveness: float,
        accountability: float,
        victim_mentality: float
    ) -> str:
        """Generate psychological profile summary"""
        summary = f"Psychological Profile: {name}\n\n"

        summary += f"Baseline Stress Level: {stress:.0f}/100\n"

        if gaslighting:
            summary += f"Gaslighting Indicators: {len(gaslighting)} detected\n"

        if guilt_tripping:
            summary += f"Guilt-Tripping Indicators: {len(guilt_tripping)} detected\n"

        summary += f"Defensiveness: {defensiveness*100:.0f}%\n"
        summary += f"Accountability: {accountability*100:.0f}%\n"
        summary += f"Victim Mentality: {victim_mentality*100:.0f}%\n"

        return summary

    def _identify_risk_indicators(
        self,
        gaslighting: List[str],
        guilt_tripping: List[str],
        threatening: List[str],
        isolation: List[str],
        defensiveness: float,
        victim_mentality: float
    ) -> List[str]:
        """Identify risk indicators"""
        indicators = []

        if gaslighting:
            indicators.append("Gaslighting behavior detected")
        if guilt_tripping:
            indicators.append("Guilt-tripping behavior detected")
        if threatening:
            indicators.append("Threatening language detected")
        if isolation:
            indicators.append("Isolation tactics detected")
        if defensiveness > 0.7:
            indicators.append("High defensiveness")
        if victim_mentality > 0.6:
            indicators.append("Strong victim mentality")

        return indicators

    def _identify_red_flags(
        self,
        stress_spikes: int,
        escalation_rate: float,
        consistency: float,
        accountability: float
    ) -> List[str]:
        """Identify behavioral red flags"""
        flags = []

        if stress_spikes > 3:
            flags.append(f"Multiple stress spikes ({stress_spikes})")
        if escalation_rate > 5:
            flags.append("Rapid emotional escalation")
        if consistency < 0.5:
            flags.append("Low emotional consistency")
        if accountability < 0.2:
            flags.append("Very low accountability")

        return flags

    def compare_profiles(
        self,
        participant_a_id: str,
        participant_b_id: str
    ) -> Optional[ComparativeAnalysis]:
        """
        Compare two psychological profiles

        Args:
            participant_a_id: First participant ID
            participant_b_id: Second participant ID

        Returns:
            Comparative analysis
        """
        profile_a = self.profiles.get(participant_a_id)
        profile_b = self.profiles.get(participant_b_id)

        if not profile_a or not profile_b:
            return None

        # Calculate differences
        stress_diff = profile_a.baseline_stress - profile_b.baseline_stress
        defensiveness_diff = profile_a.defensiveness_score - profile_b.defensiveness_score
        accountability_diff = profile_a.accountability_score - profile_b.accountability_score
        victim_diff = profile_a.victim_mentality_score - profile_b.victim_mentality_score
        consistency_diff = profile_a.consistency_score - profile_b.consistency_score

        # Determine patterns
        stress_pattern = "A more stressed" if stress_diff > 0 else "B more stressed"
        emotional_pattern = "A more escalating" if profile_a.emotional_escalation_rate > profile_b.emotional_escalation_rate else "B more escalating"

        # Key differences
        key_differences = []
        if abs(defensiveness_diff) > 0.3:
            key_differences.append(f"Defensiveness difference: {abs(defensiveness_diff)*100:.0f}%")
        if abs(accountability_diff) > 0.3:
            key_differences.append(f"Accountability difference: {abs(accountability_diff)*100:.0f}%")
        if abs(victim_diff) > 0.3:
            key_differences.append(f"Victim mentality difference: {abs(victim_diff)*100:.0f}%")

        # Risk assessment
        risk_a = "High" if len(profile_a.risk_indicators) > 3 else "Moderate" if len(profile_a.risk_indicators) > 1 else "Low"
        risk_b = "High" if len(profile_b.risk_indicators) > 3 else "Moderate" if len(profile_b.risk_indicators) > 1 else "Low"

        comparison = ComparativeAnalysis(
            participant_a_id=participant_a_id,
            participant_b_id=participant_b_id,
            participant_a_name=profile_a.participant_name,
            participant_b_name=profile_b.participant_name,
            stress_baseline_difference=stress_diff,
            stress_response_pattern_difference=stress_pattern,
            manipulation_tactics_a=profile_a.gaslighting_indicators + profile_a.guilt_tripping_indicators,
            manipulation_tactics_b=profile_b.gaslighting_indicators + profile_b.guilt_tripping_indicators,
            unique_tactics_a=[t for t in profile_a.gaslighting_indicators if t not in profile_b.gaslighting_indicators],
            unique_tactics_b=[t for t in profile_b.gaslighting_indicators if t not in profile_a.gaslighting_indicators],
            emotional_stability_a=profile_a.emotional_consistency,
            emotional_stability_b=profile_b.emotional_consistency,
            emotional_pattern_difference=emotional_pattern,
            defensiveness_difference=defensiveness_diff,
            accountability_difference=accountability_diff,
            victim_mentality_difference=victim_diff,
            consistency_difference=consistency_diff,
            key_differences=key_differences,
            behavioral_contrast=f"{profile_a.participant_name} shows {abs(defensiveness_diff)*100:.0f}% more defensiveness than {profile_b.participant_name}",
            risk_assessment={
                profile_a.participant_name: risk_a,
                profile_b.participant_name: risk_b
            }
        )

        self.comparisons[f"{participant_a_id}_vs_{participant_b_id}"] = comparison
        return comparison

    def export_profile(self, participant_id: str, output_path: str) -> bool:
        """Export profile to file"""
        try:
            profile = self.profiles.get(participant_id)
            if not profile:
                return False

            import json
            data = {
                'participant_id': profile.participant_id,
                'participant_name': profile.participant_name,
                'baseline_stress': profile.baseline_stress,
                'stress_spike_frequency': profile.stress_spike_frequency,
                'defensiveness_score': profile.defensiveness_score,
                'accountability_score': profile.accountability_score,
                'victim_mentality_score': profile.victim_mentality_score,
                'consistency_score': profile.consistency_score,
                'psychological_profile_summary': profile.psychological_profile_summary,
                'risk_indicators': profile.risk_indicators,
                'behavioral_red_flags': profile.behavioral_red_flags,
                'gaslighting_indicators': profile.gaslighting_indicators,
                'guilt_tripping_indicators': profile.guilt_tripping_indicators,
                'threatening_language': profile.threatening_language,
                'isolation_tactics': profile.isolation_tactics
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported profile to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    analyzer = ComparativePsychologyAnalyzer()
    print("Comparative Psychology Analyzer initialized")
