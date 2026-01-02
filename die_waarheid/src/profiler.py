"""
Psychological Profiler for Die Waarheid
Advanced pattern detection and behavioral analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter

import pandas as pd

from config import (
    GASLIGHTING_PHRASES,
    TOXICITY_PHRASES,
    NARCISSISTIC_PATTERNS
)

logger = logging.getLogger(__name__)


class PsychologicalProfiler:
    """
    Advanced psychological profiling engine
    Analyzes communication patterns, behavioral indicators, and relationship dynamics
    """

    def __init__(self):
        self.message_data = []
        self.sender_profiles = {}
        self.relationship_dynamics = {}

    def add_messages(self, messages: List[Dict]) -> None:
        """
        Add messages for analysis

        Args:
            messages: List of message dictionaries
        """
        self.message_data = messages
        logger.info(f"Added {len(messages)} messages for profiling")

    def analyze_sender(self, sender: str) -> Dict:
        """
        Analyze communication patterns for a specific sender

        Args:
            sender: Sender name

        Returns:
            Dictionary with sender profile
        """
        sender_messages = [m for m in self.message_data if m.get('sender') == sender]

        if not sender_messages:
            return {'sender': sender, 'message_count': 0}

        texts = [m.get('text', '') for m in sender_messages]

        profile = {
            'sender': sender,
            'message_count': len(sender_messages),
            'total_characters': sum(len(t) for t in texts),
            'average_message_length': sum(len(t) for t in texts) / len(texts) if texts else 0,
            'longest_message': max(len(t) for t in texts) if texts else 0,
            'shortest_message': min(len(t) for t in texts) if texts else 0,
            'vocabulary_diversity': self._calculate_vocabulary_diversity(texts),
            'communication_style': self._analyze_communication_style(texts),
            'emotional_intensity': self._calculate_emotional_intensity(texts),
            'question_frequency': self._calculate_question_frequency(texts),
            'exclamation_frequency': self._calculate_exclamation_frequency(texts),
            'caps_usage': self._calculate_caps_usage(texts),
            'punctuation_patterns': self._analyze_punctuation(texts),
            'response_patterns': self._analyze_response_patterns(sender),
            'toxicity_indicators': self._count_toxicity_indicators(texts),
            'gaslighting_indicators': self._count_gaslighting_indicators(texts),
            'narcissistic_indicators': self._count_narcissistic_indicators(texts),
        }

        self.sender_profiles[sender] = profile
        logger.info(f"Analyzed profile for {sender}: {len(sender_messages)} messages")

        return profile

    def _calculate_vocabulary_diversity(self, texts: List[str]) -> float:
        """Calculate vocabulary diversity (unique words / total words)"""
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = len(set(all_words))
        total_words = len(all_words)

        return unique_words / total_words if total_words > 0 else 0.0

    def _analyze_communication_style(self, texts: List[str]) -> str:
        """Analyze overall communication style"""
        if not texts:
            return "unknown"

        avg_length = sum(len(t) for t in texts) / len(texts)
        question_count = sum(t.count('?') for t in texts)
        exclamation_count = sum(t.count('!') for t in texts)

        if avg_length < 20:
            return "terse"
        elif avg_length > 100:
            return "verbose"
        elif question_count > len(texts) * 0.3:
            return "inquisitive"
        elif exclamation_count > len(texts) * 0.3:
            return "emphatic"
        else:
            return "balanced"

    def _calculate_emotional_intensity(self, texts: List[str]) -> float:
        """Calculate emotional intensity (0-1)"""
        if not texts:
            return 0.0

        intensity_markers = ['!!!', '???', '...', 'HATE', 'LOVE', 'NEVER', 'ALWAYS']
        intensity_count = 0

        for text in texts:
            for marker in intensity_markers:
                intensity_count += text.count(marker)

        return min(1.0, intensity_count / (len(texts) * 5))

    def _calculate_question_frequency(self, texts: List[str]) -> float:
        """Calculate frequency of questions"""
        if not texts:
            return 0.0

        question_count = sum(t.count('?') for t in texts)
        return question_count / len(texts) if texts else 0.0

    def _calculate_exclamation_frequency(self, texts: List[str]) -> float:
        """Calculate frequency of exclamations"""
        if not texts:
            return 0.0

        exclamation_count = sum(t.count('!') for t in texts)
        return exclamation_count / len(texts) if texts else 0.0

    def _calculate_caps_usage(self, texts: List[str]) -> float:
        """Calculate percentage of text in CAPS"""
        if not texts:
            return 0.0

        total_chars = sum(len(t) for t in texts)
        caps_chars = sum(sum(1 for c in t if c.isupper()) for t in texts)

        return caps_chars / total_chars if total_chars > 0 else 0.0

    def _analyze_punctuation(self, texts: List[str]) -> Dict:
        """Analyze punctuation patterns"""
        punctuation_counts = {
            'periods': 0,
            'commas': 0,
            'semicolons': 0,
            'colons': 0,
            'ellipsis': 0,
            'multiple_punctuation': 0
        }

        for text in texts:
            punctuation_counts['periods'] += text.count('.')
            punctuation_counts['commas'] += text.count(',')
            punctuation_counts['semicolons'] += text.count(';')
            punctuation_counts['colons'] += text.count(':')
            punctuation_counts['ellipsis'] += text.count('...')
            punctuation_counts['multiple_punctuation'] += len([c for c in text if c in '!?.' and text.count(c) > 1])

        return punctuation_counts

    def _analyze_response_patterns(self, sender: str) -> Dict:
        """Analyze response patterns (timing, length, etc.)"""
        sender_messages = [m for m in self.message_data if m.get('sender') == sender]

        if len(sender_messages) < 2:
            return {'response_count': 0}

        response_times = []
        response_lengths = []

        for i in range(1, len(sender_messages)):
            prev_time = sender_messages[i-1].get('timestamp')
            curr_time = sender_messages[i].get('timestamp')

            if prev_time and curr_time:
                time_diff = (curr_time - prev_time).total_seconds()
                if 0 < time_diff < 86400:
                    response_times.append(time_diff)

            response_lengths.append(len(sender_messages[i].get('text', '')))

        return {
            'response_count': len(sender_messages) - 1,
            'avg_response_time_seconds': sum(response_times) / len(response_times) if response_times else 0,
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0
        }

    def _count_toxicity_indicators(self, texts: List[str]) -> int:
        """Count toxicity indicators in texts"""
        count = 0
        for text in texts:
            text_lower = text.lower()
            for phrase in TOXICITY_PHRASES:
                count += text_lower.count(phrase.lower())

        return count

    def _count_gaslighting_indicators(self, texts: List[str]) -> int:
        """Count gaslighting indicators in texts"""
        count = 0
        for text in texts:
            text_lower = text.lower()
            for phrase in GASLIGHTING_PHRASES:
                count += text_lower.count(phrase.lower())

        return count

    def _count_narcissistic_indicators(self, texts: List[str]) -> int:
        """Count narcissistic indicators in texts"""
        count = 0
        for text in texts:
            text_lower = text.lower()
            for pattern in NARCISSISTIC_PATTERNS:
                count += text_lower.count(pattern.lower())

        return count

    def analyze_relationship_dynamics(self) -> Dict:
        """
        Analyze relationship dynamics between participants

        Returns:
            Dictionary with relationship analysis
        """
        if not self.message_data:
            return {}

        senders = set(m.get('sender') for m in self.message_data)

        dynamics = {
            'participants': list(senders),
            'total_messages': len(self.message_data),
            'message_distribution': {},
            'interaction_patterns': {},
            'power_indicators': {},
            'conflict_indicators': {}
        }

        for sender in senders:
            sender_messages = [m for m in self.message_data if m.get('sender') == sender]
            dynamics['message_distribution'][sender] = len(sender_messages)

        if len(senders) == 2:
            senders_list = list(senders)
            sender1, sender2 = senders_list[0], senders_list[1]

            sender1_msgs = [m for m in self.message_data if m.get('sender') == sender1]
            sender2_msgs = [m for m in self.message_data if m.get('sender') == sender2]

            dynamics['interaction_patterns'] = {
                'sender1': sender1,
                'sender1_message_count': len(sender1_msgs),
                'sender1_avg_length': sum(len(m.get('text', '')) for m in sender1_msgs) / len(sender1_msgs) if sender1_msgs else 0,
                'sender2': sender2,
                'sender2_message_count': len(sender2_msgs),
                'sender2_avg_length': sum(len(m.get('text', '')) for m in sender2_msgs) / len(sender2_msgs) if sender2_msgs else 0,
                'message_ratio': len(sender1_msgs) / len(sender2_msgs) if sender2_msgs else 0
            }

            sender1_toxicity = sum(1 for m in sender1_msgs if self._has_toxicity(m.get('text', '')))
            sender2_toxicity = sum(1 for m in sender2_msgs if self._has_toxicity(m.get('text', '')))

            dynamics['conflict_indicators'] = {
                'sender1_toxicity_messages': sender1_toxicity,
                'sender2_toxicity_messages': sender2_toxicity,
                'overall_conflict_level': min(1.0, (sender1_toxicity + sender2_toxicity) / max(len(self.message_data), 1))
            }

        logger.info(f"Analyzed relationship dynamics for {len(senders)} participants")
        self.relationship_dynamics = dynamics

        return dynamics

    def _has_toxicity(self, text: str) -> bool:
        """Check if text contains toxicity indicators"""
        text_lower = text.lower()
        for phrase in TOXICITY_PHRASES:
            if phrase.lower() in text_lower:
                return True

        return False

    def get_all_profiles(self) -> Dict:
        """
        Get all analyzed sender profiles

        Returns:
            Dictionary with all profiles
        """
        return self.sender_profiles

    def get_profile(self, sender: str) -> Optional[Dict]:
        """
        Get profile for specific sender

        Args:
            sender: Sender name

        Returns:
            Profile dictionary or None
        """
        if sender not in self.sender_profiles:
            return self.analyze_sender(sender)

        return self.sender_profiles.get(sender)

    def generate_profile_summary(self, sender: str) -> str:
        """
        Generate human-readable profile summary

        Args:
            sender: Sender name

        Returns:
            Formatted profile summary
        """
        profile = self.get_profile(sender)

        if not profile or profile.get('message_count', 0) == 0:
            return f"No profile data for {sender}"

        summary = f"""
## Profile: {sender}

**Communication Overview:**
- Total Messages: {profile.get('message_count', 0)}
- Average Message Length: {profile.get('average_message_length', 0):.0f} characters
- Communication Style: {profile.get('communication_style', 'unknown')}

**Linguistic Patterns:**
- Vocabulary Diversity: {profile.get('vocabulary_diversity', 0):.2%}
- Question Frequency: {profile.get('question_frequency', 0):.2f} per message
- Exclamation Frequency: {profile.get('exclamation_frequency', 0):.2f} per message
- CAPS Usage: {profile.get('caps_usage', 0):.2%}
- Emotional Intensity: {profile.get('emotional_intensity', 0):.2f}/1.0

**Behavioral Indicators:**
- Toxicity Indicators: {profile.get('toxicity_indicators', 0)}
- Gaslighting Indicators: {profile.get('gaslighting_indicators', 0)}
- Narcissistic Indicators: {profile.get('narcissistic_indicators', 0)}

**Response Patterns:**
- Average Response Time: {profile.get('response_patterns', {}).get('avg_response_time_seconds', 0):.0f}s
- Average Response Length: {profile.get('response_patterns', {}).get('avg_response_length', 0):.0f} characters
"""

        return summary.strip()

    def export_profiles_to_dataframe(self) -> pd.DataFrame:
        """
        Export all profiles to pandas DataFrame

        Returns:
            DataFrame with profile data
        """
        if not self.sender_profiles:
            return pd.DataFrame()

        profiles_list = []

        for sender, profile in self.sender_profiles.items():
            row = {
                'sender': sender,
                'message_count': profile.get('message_count', 0),
                'avg_message_length': profile.get('average_message_length', 0),
                'communication_style': profile.get('communication_style', ''),
                'vocabulary_diversity': profile.get('vocabulary_diversity', 0),
                'emotional_intensity': profile.get('emotional_intensity', 0),
                'toxicity_indicators': profile.get('toxicity_indicators', 0),
                'gaslighting_indicators': profile.get('gaslighting_indicators', 0),
                'narcissistic_indicators': profile.get('narcissistic_indicators', 0),
            }

            profiles_list.append(row)

        return pd.DataFrame(profiles_list)


if __name__ == "__main__":
    profiler = PsychologicalProfiler()
    print("Psychological Profiler initialized")
