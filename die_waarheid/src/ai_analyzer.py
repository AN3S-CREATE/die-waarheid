"""
AI Analyzer for Die Waarheid
Gemini-powered psychological profiling and pattern detection
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from functools import wraps

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    import warnings
    warnings.filterwarnings("ignore", message=".*google.generativeai.*deprecated.*")
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    GEMINI_SAFETY_SETTINGS,
    GASLIGHTING_PHRASES,
    TOXICITY_PHRASES,
    NARCISSISTIC_PATTERNS
)

logger = logging.getLogger(__name__)


def rate_limit(calls_per_minute: int = 30):
    """Decorator for rate limiting API calls"""
    min_interval = 60.0 / calls_per_minute
    last_call = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 2.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"Attempt {attempt} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator


class AIAnalyzer:
    """
    AI-powered analysis using Google Gemini
    Performs psychological profiling, contradiction detection, and pattern matching
    """

    def __init__(self):
        self.configured = False
        self.model = None
        self.configure_gemini()

    def sanitize_input(self, text: str, max_length: int = 10000) -> str:
        """
        Sanitize user input for AI prompts to prevent injection attacks
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        text = text[:max_length]
        text = text.replace('```', '')
        text = text.replace('"""', '')
        text = text.replace("'''", '')
        text = text.strip()
        
        return text

    def configure_gemini(self) -> bool:
        """
        Configure Gemini API

        Returns:
            True if configuration successful
        """
        try:
            if not GENAI_AVAILABLE:
                logger.warning("google.generativeai not installed; AIAnalyzer disabled")
                return False

            if not GEMINI_API_KEY or GEMINI_API_KEY == "placeholder_gemini_key":
                logger.warning("GEMINI_API_KEY not set or using placeholder value. AI analysis will be disabled.")
                return False

            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                safety_settings=GEMINI_SAFETY_SETTINGS
            )
            self.configured = True
            logger.info(f"Configured Gemini model: {GEMINI_MODEL}")
            return True

        except Exception as e:
            logger.error(f"Error configuring Gemini: {str(e)}")
            return False

    @rate_limit(calls_per_minute=30)
    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def analyze_message(self, text: str) -> Dict:
        """
        Analyze a single message for toxicity, emotion, and patterns

        Args:
            text: Message text to analyze

        Returns:
            Dictionary with analysis results
        """
        if not self.configured:
            logger.warning("Gemini not configured")
            return {
                'success': False,
                'message': 'Gemini not configured',
                'text': text
            }

        try:
            text = self.sanitize_input(text)
            
            prompt = f"""Analyze this message for psychological indicators:

Message: "{text}"

Provide analysis in JSON format with:
1. emotion: (positive/negative/neutral/mixed)
2. toxicity_score: (0-1, where 1 is most toxic)
3. aggression_level: (low/medium/high)
4. confidence: (0-1)

Be concise and return only valid JSON."""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )

            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]

            import json
            analysis = json.loads(result_text)

            return {
                'success': True,
                'text': text,
                'emotion': analysis.get('emotion', 'unknown'),
                'toxicity_score': float(analysis.get('toxicity_score', 0)),
                'aggression_level': analysis.get('aggression_level', 'low'),
                'confidence': float(analysis.get('confidence', 0))
            }

        except Exception as e:
            logger.error(f"Error analyzing message: {str(e)}")
            return {
                'success': False,
                'message': f'Analysis error: {str(e)}',
                'text': text
            }

    def detect_gaslighting(self, text: str) -> Dict:
        """
        Detect gaslighting patterns in text

        Args:
            text: Message text to analyze

        Returns:
            Dictionary with gaslighting detection results
        """
        detected_phrases = []
        text_lower = text.lower()

        for phrase in GASLIGHTING_PHRASES:
            if phrase.lower() in text_lower:
                detected_phrases.append(phrase)

        gaslighting_score = min(1.0, len(detected_phrases) / 3.0)

        return {
            'gaslighting_detected': gaslighting_score > 0.3,
            'gaslighting_score': gaslighting_score,
            'detected_phrases': detected_phrases,
            'confidence': 0.8 if detected_phrases else 0.1
        }

    def detect_toxicity(self, text: str) -> Dict:
        """
        Detect toxic language patterns

        Args:
            text: Message text to analyze

        Returns:
            Dictionary with toxicity detection results
        """
        detected_phrases = []
        text_lower = text.lower()

        for phrase in TOXICITY_PHRASES:
            if phrase.lower() in text_lower:
                detected_phrases.append(phrase)

        toxicity_score = min(1.0, len(detected_phrases) / 5.0)

        return {
            'toxicity_detected': toxicity_score > 0.3,
            'toxicity_score': toxicity_score,
            'detected_phrases': detected_phrases,
            'confidence': 0.8 if detected_phrases else 0.1
        }

    def detect_narcissistic_patterns(self, text: str) -> Dict:
        """
        Detect narcissistic behavior patterns

        Args:
            text: Message text to analyze

        Returns:
            Dictionary with narcissistic pattern detection results
        """
        detected_patterns = []
        text_lower = text.lower()

        for pattern in NARCISSISTIC_PATTERNS:
            if pattern.lower() in text_lower:
                detected_patterns.append(pattern)

        narcissism_score = min(1.0, len(detected_patterns) / 3.0)

        return {
            'narcissistic_patterns_detected': narcissism_score > 0.3,
            'narcissism_score': narcissism_score,
            'detected_patterns': detected_patterns,
            'confidence': 0.7 if detected_patterns else 0.1
        }

    @rate_limit(calls_per_minute=30)
    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def analyze_conversation(self, messages: List[Dict]) -> Dict:
        """
        Analyze entire conversation for patterns and dynamics

        Args:
            messages: List of message dictionaries with 'sender' and 'text' keys

        Returns:
            Dictionary with conversation analysis
        """
        if not messages:
            return {'success': False, 'message': 'No messages to analyze'}

        if not self.configured:
            logger.warning("Gemini not configured")
            return {
                'success': False,
                'message': 'AI analysis not available - Gemini not configured',
                'overall_tone': 'unknown',
                'power_dynamics': 'unknown',
                'communication_style': 'unknown',
                'conflict_level': 0,
                'manipulation_indicators': [],
                'summary': 'AI analysis unavailable'
            }

        try:
            conversation_text = "\n".join([
                f"{m.get('sender', 'Unknown')}: {self.sanitize_input(m.get('text', ''))}"
                for m in messages
            ])

            prompt = f"""Analyze this conversation for psychological dynamics and patterns:

{conversation_text}

Provide analysis in JSON format with:
1. overall_tone: (positive/negative/neutral/mixed)
2. power_dynamics: (balanced/one_sided/abusive)
3. communication_style: (direct/indirect/passive_aggressive)
4. conflict_level: (0-1, where 1 is high conflict)
5. manipulation_indicators: (list of detected patterns)
6. summary: (brief 1-2 sentence summary)

Return only valid JSON."""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )

            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]

            import json
            analysis = json.loads(result_text)

            return {
                'success': True,
                'total_messages': len(messages),
                'overall_tone': analysis.get('overall_tone', 'unknown'),
                'power_dynamics': analysis.get('power_dynamics', 'unknown'),
                'communication_style': analysis.get('communication_style', 'unknown'),
                'conflict_level': float(analysis.get('conflict_level', 0)),
                'manipulation_indicators': analysis.get('manipulation_indicators', []),
                'summary': analysis.get('summary', '')
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")
            return {
                'success': False,
                'message': f'Analysis error: {str(e)}'
            }

    @rate_limit(calls_per_minute=30)
    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def detect_contradictions(self, messages: List[Dict]) -> Dict:
        """
        Detect contradictions and inconsistencies in conversation

        Args:
            messages: List of message dictionaries

        Returns:
            Dictionary with detected contradictions
        """
        if not messages:
            return {'success': False, 'contradictions': []}

        try:
            conversation_text = "\n".join([
                f"{m.get('sender', 'Unknown')}: {self.sanitize_input(m.get('text', ''))}"
                for m in messages
            ])

            prompt = f"""Identify contradictions and inconsistencies in this conversation:

{conversation_text}

List any statements that contradict earlier claims or are logically inconsistent.
Return JSON with:
1. contradictions: (list of objects with 'statement1', 'statement2', 'explanation')
2. inconsistency_score: (0-1)
3. reliability_assessment: (high/medium/low)

Return only valid JSON."""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )

            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]

            import json
            analysis = json.loads(result_text)

            return {
                'success': True,
                'contradictions': analysis.get('contradictions', []),
                'inconsistency_score': float(analysis.get('inconsistency_score', 0)),
                'reliability_assessment': analysis.get('reliability_assessment', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error detecting contradictions: {str(e)}")
            return {
                'success': False,
                'message': f'Analysis error: {str(e)}',
                'contradictions': []
            }

    @rate_limit(calls_per_minute=30)
    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def generate_psychological_profile(self, messages: List[Dict], forensics_data: Optional[List[Dict]] = None) -> Dict:
        """
        Generate comprehensive psychological profile

        Args:
            messages: List of message dictionaries
            forensics_data: Optional list of forensic analysis results

        Returns:
            Dictionary with psychological profile
        """
        if not messages:
            return {'success': False, 'message': 'No messages to analyze'}

        if not self.configured:
            logger.warning("Gemini not configured")
            return {
                'success': False,
                'message': 'AI analysis not available - Gemini not configured',
                'personality_traits': [],
                'behavioral_patterns': [],
                'risk_factors': [],
                'communication_style': 'unknown',
                'psychological_state': 'unknown',
                'recommendations': ['AI analysis unavailable - configure Gemini API key']
            }

        try:
            conversation_text = "\n".join([
                f"{m.get('sender', 'Unknown')}: {self.sanitize_input(m.get('text', ''))}"
                for m in messages
            ])

            forensics_context = ""
            if forensics_data:
                avg_stress = sum(f.get('stress_level', 0) for f in forensics_data) / len(forensics_data)
                forensics_context = f"\n\nAudio Analysis Context:\n- Average stress level: {avg_stress:.2f}/100"

            prompt = f"""Create a psychological profile based on this conversation:

{conversation_text}{forensics_context}

Analyze and provide in JSON format:
1. personality_traits: (list of observed traits)
2. communication_patterns: (list of patterns)
3. emotional_regulation: (low/moderate/high)
4. stress_indicators: (list of indicators)
5. relationship_dynamics: (description)
6. risk_assessment: (low/medium/high)
7. recommendations: (list of observations)

Return only valid JSON."""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )

            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]

            import json
            profile = json.loads(result_text)

            return {
                'success': True,
                'personality_traits': profile.get('personality_traits', []),
                'communication_patterns': profile.get('communication_patterns', []),
                'emotional_regulation': profile.get('emotional_regulation', 'unknown'),
                'stress_indicators': profile.get('stress_indicators', []),
                'relationship_dynamics': profile.get('relationship_dynamics', ''),
                'risk_assessment': profile.get('risk_assessment', 'unknown'),
                'recommendations': profile.get('recommendations', [])
            }

        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}")
            return {
                'success': False,
                'message': f'Profile generation error: {str(e)}'
            }

    def calculate_trust_score(
        self,
        conversation_analysis: Dict,
        contradiction_analysis: Dict,
        toxicity_score: float,
        gaslighting_score: float
    ) -> float:
        """
        Calculate composite trust score

        Args:
            conversation_analysis: Conversation analysis results
            contradiction_analysis: Contradiction detection results
            toxicity_score: Toxicity score (0-1)
            gaslighting_score: Gaslighting score (0-1)

        Returns:
            Trust score (0-100)
        """
        try:
            base_score = 100.0

            conflict_penalty = conversation_analysis.get('conflict_level', 0) * 30
            inconsistency_penalty = contradiction_analysis.get('inconsistency_score', 0) * 25
            toxicity_penalty = toxicity_score * 20
            gaslighting_penalty = gaslighting_score * 25

            trust_score = base_score - conflict_penalty - inconsistency_penalty - toxicity_penalty - gaslighting_penalty

            return max(0, min(100, trust_score))

        except Exception as e:
            logger.error(f"Error calculating trust score: {str(e)}")
            return 50.0

    def get_analysis_status(self) -> Dict:
        """
        Get current analysis status

        Returns:
            Dictionary with status information
        """
        return {
            'configured': self.configured,
            'model': GEMINI_MODEL,
            'temperature': GEMINI_TEMPERATURE,
            'max_tokens': GEMINI_MAX_TOKENS
        }


if __name__ == "__main__":
    analyzer = AIAnalyzer()
    print(f"AI Analyzer Status: {analyzer.get_analysis_status()}")
