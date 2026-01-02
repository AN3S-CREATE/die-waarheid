"""
Multi-Language Support Extension for Die Waarheid
Extends Afrikaans verification to multiple languages
Handles code-switching detection and accent analysis

SUPPORTED LANGUAGES:
- English (primary)
- Afrikaans (secondary)
- Code-switching detection (mixing languages)
- Accent analysis for authenticity
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    ENGLISH = "english"
    AFRIKAANS = "afrikaans"
    MIXED = "mixed"


class LanguageConfidence(Enum):
    """Confidence levels for language detection"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class LanguageDetection:
    """Language detection result"""
    detected_language: Language
    confidence: float
    primary_language: Language
    secondary_language: Optional[Language]
    code_switching_detected: bool
    code_switch_points: List[Dict[str, Any]]
    accent_markers: List[str]
    authenticity_score: float


@dataclass
class LanguageAnalysis:
    """Complete language analysis"""
    text_id: str
    original_text: str
    
    # Detection
    language_detection: LanguageDetection
    
    # Authenticity
    native_speaker_indicators: List[str]
    non_native_indicators: List[str]
    authenticity_confidence: float
    
    # Code-switching
    code_switch_frequency: float
    code_switch_patterns: List[str]
    code_switch_reason: Optional[str]
    
    # Accent analysis
    accent_detected: Optional[str]
    accent_confidence: float
    
    # Recommendations
    authenticity_assessment: str
    recommendations: List[str]


class MultilingualAnalyzer:
    """
    Analyzes text in multiple languages
    Detects code-switching and accent patterns
    """

    def __init__(self):
        """Initialize analyzer"""
        self.analysis_counter = 0
        self.language_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Language-specific word lists
        self.english_words = self._load_english_words()
        self.afrikaans_words = self._load_afrikaans_words()
        
        # Code-switching indicators
        self.code_switch_patterns = [
            r'\b(ja|nee|dankie|asseblief)\b',  # Afrikaans in English
            r'\b(yes|no|thank|please)\b',  # English in Afrikaans
        ]

    def _load_english_words(self) -> set:
        """Load common English words"""
        return {
            'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'cannot', 'not', 'no', 'yes', 'hello', 'goodbye'
        }

    def _load_afrikaans_words(self) -> set:
        """Load common Afrikaans words"""
        return {
            'die', 'en', 'of', 'maar', 'in', 'op', 'aan', 'na', 'vir',
            'is', 'are', 'was', 'were', 'wees', 'gewees',
            'het', 'hê', 'had', 'doen', 'doen', 'gaan',
            'sal', 'sou', 'moet', 'kan', 'mag', 'wil',
            'nee', 'ja', 'dankie', 'asseblief', 'hallo', 'totsiens'
        }

    def analyze_text(
        self,
        text_id: str,
        text: str,
        speaker_id: Optional[str] = None
    ) -> LanguageAnalysis:
        """
        Analyze text for language and authenticity

        Args:
            text_id: Text identifier
            text: Text to analyze
            speaker_id: Speaker identifier for profile building

        Returns:
            Language analysis
        """
        logger.info(f"Analyzing language for {text_id}")

        self.analysis_counter += 1

        # Detect language
        language_detection = self._detect_language(text)

        # Analyze authenticity
        native_indicators, non_native_indicators, authenticity_conf = self._analyze_authenticity(
            text, language_detection.primary_language
        )

        # Analyze code-switching
        code_switch_freq, code_switch_patterns, code_switch_reason = self._analyze_code_switching(
            text, language_detection
        )

        # Analyze accent
        accent_detected, accent_conf = self._analyze_accent(text, language_detection)

        # Generate assessment
        authenticity_assessment = self._generate_authenticity_assessment(
            authenticity_conf, native_indicators, non_native_indicators
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            language_detection, authenticity_conf, code_switch_freq
        )

        analysis = LanguageAnalysis(
            text_id=text_id,
            original_text=text,
            language_detection=language_detection,
            native_speaker_indicators=native_indicators,
            non_native_indicators=non_native_indicators,
            authenticity_confidence=authenticity_conf,
            code_switch_frequency=code_switch_freq,
            code_switch_patterns=code_switch_patterns,
            code_switch_reason=code_switch_reason,
            accent_detected=accent_detected,
            accent_confidence=accent_conf,
            authenticity_assessment=authenticity_assessment,
            recommendations=recommendations
        )

        # Update speaker profile
        if speaker_id:
            self._update_speaker_profile(speaker_id, analysis)

        return analysis

    def _detect_language(self, text: str) -> LanguageDetection:
        """Detect language in text"""
        text_lower = text.lower()
        words = text_lower.split()

        english_count = sum(1 for word in words if word in self.english_words)
        afrikaans_count = sum(1 for word in words if word in self.afrikaans_words)

        total_recognized = english_count + afrikaans_count
        
        if total_recognized == 0:
            return LanguageDetection(
                detected_language=Language.ENGLISH,
                confidence=0.5,
                primary_language=Language.ENGLISH,
                secondary_language=None,
                code_switching_detected=False,
                code_switch_points=[],
                accent_markers=[],
                authenticity_score=0.5
            )

        english_ratio = english_count / total_recognized
        afrikaans_ratio = afrikaans_count / total_recognized

        # Determine primary and secondary languages
        if english_ratio > 0.7:
            primary = Language.ENGLISH
            secondary = Language.AFRIKAANS if afrikaans_ratio > 0.1 else None
            detected = Language.ENGLISH
            confidence = english_ratio
        elif afrikaans_ratio > 0.7:
            primary = Language.AFRIKAANS
            secondary = Language.ENGLISH if english_ratio > 0.1 else None
            detected = Language.AFRIKAANS
            confidence = afrikaans_ratio
        else:
            primary = Language.ENGLISH if english_ratio > afrikaans_ratio else Language.AFRIKAANS
            secondary = Language.AFRIKAANS if english_ratio > afrikaans_ratio else Language.ENGLISH
            detected = Language.MIXED
            confidence = max(english_ratio, afrikaans_ratio)

        # Detect code-switching points
        code_switch_points = self._find_code_switch_points(text, primary)
        code_switching = len(code_switch_points) > 0

        # Detect accent markers
        accent_markers = self._detect_accent_markers(text, primary)

        return LanguageDetection(
            detected_language=detected,
            confidence=confidence,
            primary_language=primary,
            secondary_language=secondary,
            code_switching_detected=code_switching,
            code_switch_points=code_switch_points,
            accent_markers=accent_markers,
            authenticity_score=confidence
        )

    def _find_code_switch_points(self, text: str, primary_language: Language) -> List[Dict[str, Any]]:
        """Find points where code-switching occurs"""
        import re
        
        code_switch_points = []
        
        if primary_language == Language.ENGLISH:
            afrikaans_patterns = [
                (r'\bja\b', 'Afrikaans: yes'),
                (r'\bnee\b', 'Afrikaans: no'),
                (r'\bdankie\b', 'Afrikaans: thank you'),
                (r'\basseblief\b', 'Afrikaans: please'),
            ]
            
            for pattern, meaning in afrikaans_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    code_switch_points.append({
                        'position': match.start(),
                        'word': match.group(),
                        'language': 'Afrikaans',
                        'meaning': meaning
                    })
        
        elif primary_language == Language.AFRIKAANS:
            english_patterns = [
                (r'\byes\b', 'English: yes'),
                (r'\bno\b', 'English: no'),
                (r'\bthank\b', 'English: thank'),
                (r'\bplease\b', 'English: please'),
            ]
            
            for pattern, meaning in english_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    code_switch_points.append({
                        'position': match.start(),
                        'word': match.group(),
                        'language': 'English',
                        'meaning': meaning
                    })

        return code_switch_points

    def _detect_accent_markers(self, text: str, language: Language) -> List[str]:
        """Detect accent markers in text"""
        markers = []

        if language == Language.AFRIKAANS:
            # Afrikaans-specific accent markers
            if 'jy' in text.lower():
                markers.append("Use of 'jy' (informal you)")
            if 'u' in text.lower():
                markers.append("Use of 'u' (formal you)")
            if 'gaan' in text.lower():
                markers.append("Use of 'gaan' (going)")

        elif language == Language.ENGLISH:
            # English accent markers
            if 'gonna' in text.lower():
                markers.append("Colloquial: 'gonna'")
            if 'wanna' in text.lower():
                markers.append("Colloquial: 'wanna'")
            if 'innit' in text.lower():
                markers.append("British slang: 'innit'")

        return markers

    def _analyze_authenticity(
        self,
        text: str,
        language: Language
    ) -> Tuple[List[str], List[str], float]:
        """Analyze authenticity of language"""
        native_indicators = []
        non_native_indicators = []

        text_lower = text.lower()

        if language == Language.AFRIKAANS:
            # Native Afrikaans indicators
            if 'jy' in text_lower or 'u' in text_lower:
                native_indicators.append("Proper pronoun usage")
            if 'het' in text_lower or 'is' in text_lower:
                native_indicators.append("Correct verb conjugation")

            # Non-native indicators
            if 'the' in text_lower:
                non_native_indicators.append("English article 'the' in Afrikaans context")
            if 'and' in text_lower and 'en' not in text_lower:
                non_native_indicators.append("English 'and' instead of Afrikaans 'en'")

        elif language == Language.ENGLISH:
            # Native English indicators
            if 'the' in text_lower:
                native_indicators.append("Proper article usage")
            if 'is' in text_lower or 'are' in text_lower:
                native_indicators.append("Correct verb conjugation")

            # Non-native indicators
            if 'ja' in text_lower or 'nee' in text_lower:
                non_native_indicators.append("Afrikaans words in English context")

        # Calculate authenticity confidence
        total_indicators = len(native_indicators) + len(non_native_indicators)
        if total_indicators == 0:
            authenticity_conf = 0.5
        else:
            authenticity_conf = len(native_indicators) / total_indicators

        return native_indicators, non_native_indicators, authenticity_conf

    def _analyze_code_switching(
        self,
        text: str,
        language_detection: LanguageDetection
    ) -> Tuple[float, List[str], Optional[str]]:
        """Analyze code-switching patterns"""
        if not language_detection.code_switching_detected:
            return 0.0, [], None

        code_switch_freq = len(language_detection.code_switch_points) / max(len(text.split()), 1)

        patterns = []
        for point in language_detection.code_switch_points:
            patterns.append(f"{point['word']} ({point['language']})")

        # Determine reason for code-switching
        reason = None
        if code_switch_freq < 0.05:
            reason = "Minimal code-switching - likely natural bilingual behavior"
        elif code_switch_freq < 0.15:
            reason = "Moderate code-switching - common in bilingual speakers"
        else:
            reason = "Frequent code-switching - may indicate language confusion or deliberate mixing"

        return code_switch_freq, patterns, reason

    def _analyze_accent(self, text: str, language_detection: LanguageDetection) -> Tuple[Optional[str], float]:
        """Analyze accent from text"""
        accent = None
        confidence = 0.0

        # Simple accent detection based on markers
        if language_detection.accent_markers:
            if language_detection.primary_language == Language.AFRIKAANS:
                accent = "South African Afrikaans"
                confidence = 0.7
            elif language_detection.primary_language == Language.ENGLISH:
                if 'innit' in text.lower():
                    accent = "British English"
                    confidence = 0.8
                elif 'gonna' in text.lower():
                    accent = "American English"
                    confidence = 0.6

        return accent, confidence

    def _generate_authenticity_assessment(
        self,
        authenticity_conf: float,
        native_indicators: List[str],
        non_native_indicators: List[str]
    ) -> str:
        """Generate authenticity assessment"""
        if authenticity_conf > 0.8:
            assessment = "High authenticity - appears to be native speaker"
        elif authenticity_conf > 0.6:
            assessment = "Good authenticity - likely native or fluent speaker"
        elif authenticity_conf > 0.4:
            assessment = "Moderate authenticity - possible non-native speaker"
        else:
            assessment = "Low authenticity - likely non-native speaker or language confusion"

        return assessment

    def _generate_recommendations(
        self,
        language_detection: LanguageDetection,
        authenticity_conf: float,
        code_switch_freq: float
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        if authenticity_conf < 0.5:
            recommendations.append("Verify language authenticity with native speaker")

        if language_detection.code_switching_detected:
            if code_switch_freq > 0.15:
                recommendations.append("Investigate reason for frequent code-switching")
            else:
                recommendations.append("Note code-switching patterns for linguistic analysis")

        if language_detection.detected_language == Language.MIXED:
            recommendations.append("Clarify primary language for speaker")

        if language_detection.secondary_language:
            recommendations.append(f"Speaker may be bilingual - verify {language_detection.secondary_language.value}")

        return recommendations

    def _update_speaker_profile(self, speaker_id: str, analysis: LanguageAnalysis):
        """Update speaker language profile"""
        if speaker_id not in self.language_profiles:
            self.language_profiles[speaker_id] = {
                'primary_language': analysis.language_detection.primary_language.value,
                'secondary_language': analysis.language_detection.secondary_language.value if analysis.language_detection.secondary_language else None,
                'authenticity_scores': [],
                'code_switch_frequency': [],
                'accent': analysis.accent_detected
            }

        profile = self.language_profiles[speaker_id]
        profile['authenticity_scores'].append(analysis.authenticity_confidence)
        profile['code_switch_frequency'].append(analysis.code_switch_frequency)

    def get_speaker_language_profile(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get language profile for speaker"""
        if speaker_id not in self.language_profiles:
            return None

        profile = self.language_profiles[speaker_id]
        avg_authenticity = sum(profile['authenticity_scores']) / len(profile['authenticity_scores']) if profile['authenticity_scores'] else 0
        avg_code_switch = sum(profile['code_switch_frequency']) / len(profile['code_switch_frequency']) if profile['code_switch_frequency'] else 0

        return {
            'speaker_id': speaker_id,
            'primary_language': profile['primary_language'],
            'secondary_language': profile['secondary_language'],
            'average_authenticity': avg_authenticity,
            'average_code_switch_frequency': avg_code_switch,
            'accent': profile['accent'],
            'analysis_count': len(profile['authenticity_scores'])
        }

    def compare_language_profiles(self, speaker_a_id: str, speaker_b_id: str) -> Dict[str, Any]:
        """Compare language profiles of two speakers"""
        profile_a = self.get_speaker_language_profile(speaker_a_id)
        profile_b = self.get_speaker_language_profile(speaker_b_id)

        if not profile_a or not profile_b:
            return {}

        return {
            'speaker_a': profile_a,
            'speaker_b': profile_b,
            'language_match': profile_a['primary_language'] == profile_b['primary_language'],
            'authenticity_difference': abs(profile_a['average_authenticity'] - profile_b['average_authenticity']),
            'code_switch_difference': abs(profile_a['average_code_switch_frequency'] - profile_b['average_code_switch_frequency']),
            'accent_match': profile_a['accent'] == profile_b['accent']
        }

    def export_analysis(self, analysis: LanguageAnalysis, output_path: str) -> bool:
        """Export language analysis to file"""
        try:
            import json

            data = {
                'text_id': analysis.text_id,
                'detected_language': analysis.language_detection.detected_language.value,
                'primary_language': analysis.language_detection.primary_language.value,
                'secondary_language': analysis.language_detection.secondary_language.value if analysis.language_detection.secondary_language else None,
                'language_confidence': analysis.language_detection.confidence,
                'code_switching_detected': analysis.language_detection.code_switching_detected,
                'code_switch_frequency': analysis.code_switch_frequency,
                'code_switch_patterns': analysis.code_switch_patterns,
                'code_switch_reason': analysis.code_switch_reason,
                'accent_detected': analysis.accent_detected,
                'accent_confidence': analysis.accent_confidence,
                'native_speaker_indicators': analysis.native_speaker_indicators,
                'non_native_indicators': analysis.non_native_indicators,
                'authenticity_confidence': analysis.authenticity_confidence,
                'authenticity_assessment': analysis.authenticity_assessment,
                'recommendations': analysis.recommendations
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported language analysis to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    analyzer = MultilingualAnalyzer()
    print("Multilingual Support initialized")
