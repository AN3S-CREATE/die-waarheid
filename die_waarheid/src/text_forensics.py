"""
Text Forensic Analysis for Die Waarheid
Applies the same forensic analysis to text messages as audio transcriptions

ANALYSIS TYPES:
1. Pattern Change Detection - Writing style, tone, vocabulary shifts
2. Story Flow Analysis - Narrative consistency, timeline logic
3. Contradiction Detection - Conflicting statements across messages
4. Psychological Profiling - Emotional patterns, manipulation indicators
5. Behavioral Analysis - Response patterns, timing, engagement
6. Linguistic Fingerprinting - Identify speaker characteristics
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import Counter
import json

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_SAFETY_SETTINGS,
    GASLIGHTING_THRESHOLD,
    TOXICITY_THRESHOLD,
    NARCISSISTIC_PATTERN_THRESHOLD
)

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of forensic analysis"""
    PATTERN_CHANGE = "pattern_change"
    STORY_FLOW = "story_flow"
    CONTRADICTION = "contradiction"
    PSYCHOLOGICAL = "psychological"
    BEHAVIORAL = "behavioral"
    LINGUISTIC = "linguistic"


class SeverityLevel(Enum):
    """Severity of findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ForensicFinding:
    """Single forensic finding"""
    finding_id: str
    analysis_type: AnalysisType
    severity: SeverityLevel
    title: str
    description: str
    evidence: List[str]
    message_ids: List[str]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'finding_id': self.finding_id,
            'analysis_type': self.analysis_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'evidence': self.evidence,
            'message_ids': self.message_ids,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


@dataclass
class TextMessage:
    """Represents a single text message for analysis"""
    message_id: str
    timestamp: datetime
    sender: str
    text: str
    is_media: bool = False
    media_type: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'sender': self.sender,
            'text': self.text,
            'is_media': self.is_media,
            'media_type': self.media_type,
            'reply_to': self.reply_to
        }


@dataclass
class ForensicReport:
    """Complete forensic analysis report"""
    report_id: str
    generated_at: str
    message_count: int
    sender_count: int
    date_range: Dict[str, str]
    findings: List[ForensicFinding]
    summary: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at,
            'message_count': self.message_count,
            'sender_count': self.sender_count,
            'date_range': self.date_range,
            'findings': [f.to_dict() for f in self.findings],
            'summary': self.summary,
            'risk_assessment': self.risk_assessment
        }


class PatternChangeDetector:
    """
    Detect changes in writing patterns, tone, vocabulary
    """

    def __init__(self):
        """Initialize pattern detector"""
        self.finding_counter = 0

    def analyze_writing_patterns(
        self,
        messages: List[TextMessage],
        sender: str
    ) -> List[ForensicFinding]:
        """
        Analyze writing patterns for a specific sender

        Args:
            messages: List of messages
            sender: Sender to analyze

        Returns:
            List of findings
        """
        findings = []
        sender_messages = [m for m in messages if m.sender == sender]
        
        if len(sender_messages) < 5:
            return findings

        # Sort by timestamp
        sender_messages.sort(key=lambda x: x.timestamp)

        # Analyze vocabulary changes over time
        vocab_findings = self._analyze_vocabulary_changes(sender_messages)
        findings.extend(vocab_findings)

        # Analyze sentence length patterns
        length_findings = self._analyze_length_patterns(sender_messages)
        findings.extend(length_findings)

        # Analyze punctuation and emoji usage
        style_findings = self._analyze_style_changes(sender_messages)
        findings.extend(style_findings)

        # Analyze response time patterns
        timing_findings = self._analyze_timing_patterns(sender_messages, messages)
        findings.extend(timing_findings)

        return findings

    def _analyze_vocabulary_changes(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Detect vocabulary shifts over time"""
        findings = []
        
        # Split messages into time periods
        if len(messages) < 10:
            return findings

        mid_point = len(messages) // 2
        early_messages = messages[:mid_point]
        late_messages = messages[mid_point:]

        # Extract words
        early_words = Counter()
        late_words = Counter()

        for msg in early_messages:
            words = re.findall(r'\b\w+\b', msg.text.lower())
            early_words.update(words)

        for msg in late_messages:
            words = re.findall(r'\b\w+\b', msg.text.lower())
            late_words.update(words)

        # Find significant changes
        new_words = set(late_words.keys()) - set(early_words.keys())
        dropped_words = set(early_words.keys()) - set(late_words.keys())

        # Filter common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                    'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
                    'their', 'this', 'that', 'these', 'those', 'and', 'but',
                    'or', 'so', 'if', 'then', 'than', 'when', 'where', 'what',
                    'which', 'who', 'whom', 'whose', 'to', 'of', 'in', 'for',
                    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                    'ek', 'jy', 'hy', 'sy', 'ons', 'julle', 'hulle', 'is', 'het',
                    'nie', 'en', 'maar', 'of', 'as', 'wat', 'wie', 'waar', 'dit'}

        significant_new = [w for w in new_words if w not in stopwords and late_words[w] >= 3]
        significant_dropped = [w for w in dropped_words if w not in stopwords and early_words[w] >= 3]

        if len(significant_new) > 5 or len(significant_dropped) > 5:
            self.finding_counter += 1
            findings.append(ForensicFinding(
                finding_id=f"PC_{self.finding_counter:04d}",
                analysis_type=AnalysisType.PATTERN_CHANGE,
                severity=SeverityLevel.MEDIUM,
                title="Vocabulary Shift Detected",
                description=f"Significant change in vocabulary between early and late messages.",
                evidence=[
                    f"New words appearing: {', '.join(significant_new[:10])}",
                    f"Words no longer used: {', '.join(significant_dropped[:10])}"
                ],
                message_ids=[messages[0].message_id, messages[-1].message_id],
                confidence=0.75
            ))

        return findings

    def _analyze_length_patterns(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Detect changes in message length patterns"""
        findings = []
        
        if len(messages) < 10:
            return findings

        # Calculate average lengths over time
        window_size = max(5, len(messages) // 5)
        avg_lengths = []
        
        for i in range(0, len(messages), window_size):
            window = messages[i:i + window_size]
            avg_len = sum(len(m.text) for m in window) / len(window)
            avg_lengths.append(avg_len)

        # Detect significant changes
        if len(avg_lengths) >= 3:
            max_change = max(avg_lengths) / max(min(avg_lengths), 1)
            
            if max_change > 2.0:  # More than 2x change
                self.finding_counter += 1
                findings.append(ForensicFinding(
                    finding_id=f"PC_{self.finding_counter:04d}",
                    analysis_type=AnalysisType.PATTERN_CHANGE,
                    severity=SeverityLevel.LOW,
                    title="Message Length Pattern Change",
                    description=f"Average message length changed by {max_change:.1f}x over the conversation.",
                    evidence=[
                        f"Early average: {avg_lengths[0]:.0f} characters",
                        f"Late average: {avg_lengths[-1]:.0f} characters"
                    ],
                    message_ids=[],
                    confidence=0.65
                ))

        return findings

    def _analyze_style_changes(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Detect changes in punctuation and style"""
        findings = []
        
        if len(messages) < 10:
            return findings

        mid_point = len(messages) // 2
        early = messages[:mid_point]
        late = messages[mid_point:]

        # Count style indicators
        def count_style(msgs):
            exclamations = sum(m.text.count('!') for m in msgs)
            questions = sum(m.text.count('?') for m in msgs)
            caps = sum(sum(1 for c in m.text if c.isupper()) for m in msgs)
            emojis = sum(len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', m.text)) for m in msgs)
            return {
                'exclamations': exclamations / len(msgs),
                'questions': questions / len(msgs),
                'caps_ratio': caps / max(sum(len(m.text) for m in msgs), 1),
                'emojis': emojis / len(msgs)
            }

        early_style = count_style(early)
        late_style = count_style(late)

        # Check for significant changes
        changes = []
        
        if late_style['exclamations'] > early_style['exclamations'] * 2:
            changes.append("Increased use of exclamation marks")
        elif early_style['exclamations'] > late_style['exclamations'] * 2:
            changes.append("Decreased use of exclamation marks")

        if late_style['caps_ratio'] > early_style['caps_ratio'] * 2:
            changes.append("Increased use of CAPS (possible aggression)")

        if changes:
            self.finding_counter += 1
            findings.append(ForensicFinding(
                finding_id=f"PC_{self.finding_counter:04d}",
                analysis_type=AnalysisType.PATTERN_CHANGE,
                severity=SeverityLevel.MEDIUM,
                title="Writing Style Change Detected",
                description="Significant changes in writing style over time.",
                evidence=changes,
                message_ids=[],
                confidence=0.70
            ))

        return findings

    def _analyze_timing_patterns(
        self,
        sender_messages: List[TextMessage],
        all_messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Analyze response time patterns"""
        findings = []
        
        if len(sender_messages) < 5:
            return findings

        sender = sender_messages[0].sender
        response_times = []

        # Calculate response times
        all_sorted = sorted(all_messages, key=lambda x: x.timestamp)
        
        for i, msg in enumerate(all_sorted):
            if msg.sender == sender and i > 0:
                prev_msg = all_sorted[i - 1]
                if prev_msg.sender != sender:
                    response_time = (msg.timestamp - prev_msg.timestamp).total_seconds()
                    if response_time > 0 and response_time < 86400:  # Within 24 hours
                        response_times.append(response_time)

        if len(response_times) >= 5:
            avg_response = sum(response_times) / len(response_times)
            
            # Check for unusual patterns
            very_fast = sum(1 for t in response_times if t < 10)  # Under 10 seconds
            very_slow = sum(1 for t in response_times if t > 3600)  # Over 1 hour

            if very_fast > len(response_times) * 0.3:
                self.finding_counter += 1
                findings.append(ForensicFinding(
                    finding_id=f"PC_{self.finding_counter:04d}",
                    analysis_type=AnalysisType.BEHAVIORAL,
                    severity=SeverityLevel.INFO,
                    title="Rapid Response Pattern",
                    description=f"{sender} responds very quickly ({very_fast}/{len(response_times)} under 10 seconds).",
                    evidence=[
                        f"Average response time: {avg_response:.0f} seconds",
                        f"Very fast responses: {very_fast}",
                        "This could indicate high engagement or anxiety"
                    ],
                    message_ids=[],
                    confidence=0.80
                ))

        return findings


class StoryFlowAnalyzer:
    """
    Analyze narrative consistency and story flow
    """

    def __init__(self):
        """Initialize story flow analyzer"""
        self.finding_counter = 0
        self.gemini_configured = False
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini for analysis"""
        if GENAI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                self.gemini_configured = True
            except Exception as e:
                logger.error(f"Error configuring Gemini: {e}")

    def analyze_story_flow(
        self,
        messages: List[TextMessage],
        topic: str = ""
    ) -> List[ForensicFinding]:
        """
        Analyze story flow and narrative consistency

        Args:
            messages: List of messages
            topic: Topic being discussed (optional)

        Returns:
            List of findings
        """
        findings = []

        # Extract statements that tell a story
        story_messages = self._extract_narrative_messages(messages)
        
        if len(story_messages) < 3:
            return findings

        # Analyze timeline consistency
        timeline_findings = self._analyze_timeline_consistency(story_messages)
        findings.extend(timeline_findings)

        # Analyze logical flow
        logic_findings = self._analyze_logical_flow(story_messages)
        findings.extend(logic_findings)

        # Use AI for deeper analysis
        if self.gemini_configured:
            ai_findings = self._ai_story_analysis(story_messages, topic)
            findings.extend(ai_findings)

        return findings

    def _extract_narrative_messages(
        self,
        messages: List[TextMessage]
    ) -> List[TextMessage]:
        """Extract messages that contain narrative/story elements"""
        narrative_indicators = [
            r'\b(i|ek)\s+(was|het|did|said|went|saw|told|thought)\b',
            r'\b(then|daarna|toe|when|wanneer)\b',
            r'\b(because|omdat|want)\b',
            r'\b(yesterday|gister|last\s+night|vanaand|this\s+morning)\b',
            r'\b(happened|gebeur|said|gesê)\b',
        ]
        
        narrative_messages = []
        for msg in messages:
            if any(re.search(pattern, msg.text.lower()) for pattern in narrative_indicators):
                narrative_messages.append(msg)
        
        return narrative_messages

    def _analyze_timeline_consistency(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Check for timeline inconsistencies in the narrative"""
        findings = []
        
        # Extract time references
        time_refs = []
        time_patterns = [
            (r'(\d{1,2})[:\.](\d{2})', 'time'),
            (r'(yesterday|gister)', 'yesterday'),
            (r'(last\s+night|gisteraand)', 'last_night'),
            (r'(this\s+morning|vanoggend)', 'this_morning'),
            (r'(tonight|vanaand)', 'tonight'),
            (r'(\d+)\s+(hours?|ure?|minutes?|minute)\s+ago', 'relative'),
        ]

        for msg in messages:
            for pattern, ref_type in time_patterns:
                match = re.search(pattern, msg.text.lower())
                if match:
                    time_refs.append({
                        'message': msg,
                        'type': ref_type,
                        'value': match.group(0),
                        'timestamp': msg.timestamp
                    })

        # Look for potential conflicts
        # (e.g., saying "last night" in a morning message about events from the previous day)
        # This is simplified - in production would need more sophisticated analysis

        return findings

    def _analyze_logical_flow(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Analyze logical consistency of the narrative"""
        findings = []
        
        # Look for contradictory statements
        negative_patterns = [
            (r"i\s+didn'?t", r"i\s+did"),
            (r"ek\s+het\s+nie", r"ek\s+het"),
            (r"i\s+wasn'?t", r"i\s+was"),
            (r"never", r"always"),
            (r"nooit", r"altyd"),
        ]

        for msg1 in messages:
            for msg2 in messages:
                if msg1.message_id >= msg2.message_id:
                    continue
                if msg1.sender != msg2.sender:
                    continue

                text1 = msg1.text.lower()
                text2 = msg2.text.lower()

                for neg_pattern, pos_pattern in negative_patterns:
                    neg1 = re.search(neg_pattern, text1)
                    pos2 = re.search(pos_pattern, text2)
                    neg2 = re.search(neg_pattern, text2)
                    pos1 = re.search(pos_pattern, text1)

                    if (neg1 and pos2) or (pos1 and neg2):
                        # Potential contradiction - need context
                        self.finding_counter += 1
                        findings.append(ForensicFinding(
                            finding_id=f"SF_{self.finding_counter:04d}",
                            analysis_type=AnalysisType.STORY_FLOW,
                            severity=SeverityLevel.MEDIUM,
                            title="Potential Narrative Inconsistency",
                            description="Statements that may contradict each other.",
                            evidence=[
                                f"Message 1: {msg1.text[:100]}...",
                                f"Message 2: {msg2.text[:100]}..."
                            ],
                            message_ids=[msg1.message_id, msg2.message_id],
                            confidence=0.60
                        ))
                        break

        return findings

    def _ai_story_analysis(
        self,
        messages: List[TextMessage],
        topic: str
    ) -> List[ForensicFinding]:
        """Use AI to analyze story consistency"""
        findings = []
        
        if not self.gemini_configured or len(messages) < 3:
            return findings

        try:
            # Prepare conversation for analysis
            conversation_text = "\n".join([
                f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.text}"
                for m in sorted(messages, key=lambda x: x.timestamp)[:30]  # Limit for token size
            ])

            prompt = f"""Analyze this conversation for narrative consistency and story flow.

TOPIC: {topic if topic else "General conversation"}

CONVERSATION:
{conversation_text}

ANALYZE FOR:
1. Does the story flow logically?
2. Are there timeline inconsistencies?
3. Do statements contradict each other?
4. Are there gaps in the narrative that seem suspicious?
5. Does the story change between tellings?

RESPOND IN JSON:
{{
    "story_flows_logically": true/false,
    "timeline_consistent": true/false,
    "contradictions_found": [
        {{
            "statement1": "quote",
            "statement2": "quote",
            "issue": "description"
        }}
    ],
    "narrative_gaps": ["description of gaps"],
    "story_changes": ["descriptions of how story changed"],
    "overall_credibility": 0-100,
    "concerns": ["list of concerns"]
}}"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )

            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Create findings from AI analysis
                if not result.get('story_flows_logically', True):
                    self.finding_counter += 1
                    findings.append(ForensicFinding(
                        finding_id=f"SF_{self.finding_counter:04d}",
                        analysis_type=AnalysisType.STORY_FLOW,
                        severity=SeverityLevel.HIGH,
                        title="Story Flow Issues Detected (AI Analysis)",
                        description="AI analysis found the story does not flow logically.",
                        evidence=result.get('concerns', []),
                        message_ids=[],
                        confidence=0.75
                    ))

                for contradiction in result.get('contradictions_found', []):
                    self.finding_counter += 1
                    findings.append(ForensicFinding(
                        finding_id=f"SF_{self.finding_counter:04d}",
                        analysis_type=AnalysisType.CONTRADICTION,
                        severity=SeverityLevel.HIGH,
                        title="Contradiction Detected (AI Analysis)",
                        description=contradiction.get('issue', 'Contradicting statements found'),
                        evidence=[
                            f"Statement 1: {contradiction.get('statement1', '')}",
                            f"Statement 2: {contradiction.get('statement2', '')}"
                        ],
                        message_ids=[],
                        confidence=0.80
                    ))

        except Exception as e:
            logger.error(f"AI story analysis error: {e}")

        return findings


class ContradictionDetector:
    """
    Detect contradictions in statements
    """

    def __init__(self):
        """Initialize contradiction detector"""
        self.finding_counter = 0
        self.gemini_configured = False
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini"""
        if GENAI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                self.gemini_configured = True
            except Exception as e:
                logger.error(f"Error configuring Gemini: {e}")

    def detect_contradictions(
        self,
        messages: List[TextMessage],
        sender: Optional[str] = None
    ) -> List[ForensicFinding]:
        """
        Detect contradictions in messages

        Args:
            messages: List of messages
            sender: Optional specific sender to analyze

        Returns:
            List of contradiction findings
        """
        findings = []

        if sender:
            messages = [m for m in messages if m.sender == sender]

        if len(messages) < 2:
            return findings

        # Group messages by topic/subject
        topic_groups = self._group_by_topic(messages)
        
        # Check each topic group for contradictions
        for topic, topic_messages in topic_groups.items():
            if len(topic_messages) >= 2:
                topic_findings = self._check_topic_contradictions(topic, topic_messages)
                findings.extend(topic_findings)

        # AI-powered deep contradiction analysis
        if self.gemini_configured and len(messages) >= 5:
            ai_findings = self._ai_contradiction_analysis(messages)
            findings.extend(ai_findings)

        return findings

    def _group_by_topic(
        self,
        messages: List[TextMessage]
    ) -> Dict[str, List[TextMessage]]:
        """Group messages by apparent topic"""
        # Simple keyword-based grouping
        topic_keywords = {
            'location': ['was', 'went', 'there', 'here', 'home', 'work', 'place'],
            'time': ['time', 'when', 'hour', 'morning', 'night', 'yesterday', 'today'],
            'people': ['met', 'saw', 'with', 'friend', 'family', 'he', 'she', 'they'],
            'actions': ['did', 'made', 'said', 'told', 'gave', 'took'],
            'money': ['money', 'paid', 'cost', 'spent', 'bought', 'geld'],
        }

        groups = {topic: [] for topic in topic_keywords}
        groups['other'] = []

        for msg in messages:
            text_lower = msg.text.lower()
            matched = False
            
            for topic, keywords in topic_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    groups[topic].append(msg)
                    matched = True
                    break
            
            if not matched:
                groups['other'].append(msg)

        return {k: v for k, v in groups.items() if v}

    def _check_topic_contradictions(
        self,
        topic: str,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Check for contradictions within a topic"""
        findings = []
        
        # Extract factual claims
        claims = []
        for msg in messages:
            # Simple claim extraction (would be more sophisticated in production)
            sentences = re.split(r'[.!?]', msg.text)
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    claims.append({
                        'message': msg,
                        'claim': sentence.strip()
                    })

        # Compare claims for potential contradictions
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if claim1['message'].sender == claim2['message'].sender:
                    # Check for negation patterns
                    similarity = self._check_contradiction_patterns(
                        claim1['claim'],
                        claim2['claim']
                    )
                    
                    if similarity > 0.7:
                        self.finding_counter += 1
                        findings.append(ForensicFinding(
                            finding_id=f"CD_{self.finding_counter:04d}",
                            analysis_type=AnalysisType.CONTRADICTION,
                            severity=SeverityLevel.MEDIUM,
                            title=f"Potential Contradiction in {topic.title()} Topic",
                            description="Statements that may contradict each other.",
                            evidence=[
                                f"Claim 1: {claim1['claim'][:100]}",
                                f"Claim 2: {claim2['claim'][:100]}"
                            ],
                            message_ids=[
                                claim1['message'].message_id,
                                claim2['message'].message_id
                            ],
                            confidence=similarity
                        ))

        return findings

    def _check_contradiction_patterns(self, text1: str, text2: str) -> float:
        """Check if two texts contradict each other"""
        text1 = text1.lower()
        text2 = text2.lower()

        # Check for negation flip
        negation_pairs = [
            (r'\bdid\b', r'\bdid\s*n[o\']?t\b'),
            (r'\bwas\b', r'\bwas\s*n[o\']?t\b'),
            (r'\bhave\b', r'\bhave\s*n[o\']?t\b'),
            (r'\bhet\b', r'\bhet\s+nie\b'),
            (r'\bis\b', r'\bis\s+nie\b'),
            (r'\byes\b', r'\bno\b'),
            (r'\bja\b', r'\bnee\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\baltyd\b', r'\bnooit\b'),
        ]

        for pos_pattern, neg_pattern in negation_pairs:
            pos1 = bool(re.search(pos_pattern, text1))
            neg1 = bool(re.search(neg_pattern, text1))
            pos2 = bool(re.search(pos_pattern, text2))
            neg2 = bool(re.search(neg_pattern, text2))

            if (pos1 and neg2) or (neg1 and pos2):
                # Check if they're about the same subject
                words1 = set(re.findall(r'\b\w{4,}\b', text1))
                words2 = set(re.findall(r'\b\w{4,}\b', text2))
                overlap = len(words1 & words2)
                
                if overlap >= 2:
                    return 0.8

        return 0.0

    def _ai_contradiction_analysis(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Use AI for deep contradiction analysis"""
        findings = []
        
        try:
            # Group by sender
            by_sender = {}
            for msg in messages:
                if msg.sender not in by_sender:
                    by_sender[msg.sender] = []
                by_sender[msg.sender].append(msg)

            for sender, sender_messages in by_sender.items():
                if len(sender_messages) < 5:
                    continue

                # Prepare text for analysis
                statements = "\n".join([
                    f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}]: {m.text}"
                    for m in sorted(sender_messages, key=lambda x: x.timestamp)[:25]
                ])

                prompt = f"""Analyze these statements from {sender} for contradictions.

STATEMENTS:
{statements}

Find ANY statements that contradict each other. Look for:
1. Facts that change (locations, times, people present)
2. Denials of previously admitted things
3. Changed quantities or amounts
4. Different versions of the same event
5. Logical impossibilities

RESPOND IN JSON:
{{
    "contradictions": [
        {{
            "statement1": "exact quote",
            "statement1_time": "timestamp",
            "statement2": "exact quote",
            "statement2_time": "timestamp",
            "type": "type of contradiction",
            "severity": "high/medium/low",
            "explanation": "why these contradict"
        }}
    ],
    "consistency_score": 0-100
}}"""

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2000
                    )
                )

                json_match = re.search(r'\{[\s\S]*\}', response.text)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    for contradiction in result.get('contradictions', []):
                        severity_map = {
                            'high': SeverityLevel.HIGH,
                            'medium': SeverityLevel.MEDIUM,
                            'low': SeverityLevel.LOW
                        }
                        
                        self.finding_counter += 1
                        findings.append(ForensicFinding(
                            finding_id=f"CD_{self.finding_counter:04d}",
                            analysis_type=AnalysisType.CONTRADICTION,
                            severity=severity_map.get(
                                contradiction.get('severity', 'medium'),
                                SeverityLevel.MEDIUM
                            ),
                            title=f"Contradiction Detected - {sender}",
                            description=contradiction.get('explanation', ''),
                            evidence=[
                                f"[{contradiction.get('statement1_time', '')}] {contradiction.get('statement1', '')}",
                                f"[{contradiction.get('statement2_time', '')}] {contradiction.get('statement2', '')}"
                            ],
                            message_ids=[],
                            confidence=0.85
                        ))

        except Exception as e:
            logger.error(f"AI contradiction analysis error: {e}")

        return findings


class PsychologicalAnalyzer:
    """
    Psychological analysis of text messages
    Same analysis as audio but for text
    """

    def __init__(self):
        """Initialize psychological analyzer"""
        self.finding_counter = 0
        self.gemini_configured = False
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini"""
        if GENAI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                self.gemini_configured = True
            except Exception as e:
                logger.error(f"Error configuring Gemini: {e}")

    def analyze_psychological_patterns(
        self,
        messages: List[TextMessage],
        sender: Optional[str] = None
    ) -> List[ForensicFinding]:
        """
        Analyze psychological patterns in messages

        Args:
            messages: List of messages
            sender: Optional specific sender

        Returns:
            List of psychological findings
        """
        findings = []

        if sender:
            messages = [m for m in messages if m.sender == sender]

        if len(messages) < 5:
            return findings

        # Gaslighting detection
        gaslighting = self._detect_gaslighting(messages)
        findings.extend(gaslighting)

        # Manipulation patterns
        manipulation = self._detect_manipulation(messages)
        findings.extend(manipulation)

        # Emotional analysis
        emotional = self._analyze_emotional_patterns(messages)
        findings.extend(emotional)

        # AI psychological profiling
        if self.gemini_configured:
            ai_findings = self._ai_psychological_analysis(messages)
            findings.extend(ai_findings)

        return findings

    def _detect_gaslighting(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Detect gaslighting patterns"""
        findings = []
        
        gaslighting_phrases = [
            r"you'?re\s+(crazy|mad|insane|imagining|overreacting)",
            r"that\s+never\s+happened",
            r"you'?re\s+making\s+(things|that)\s+up",
            r"i\s+never\s+said\s+that",
            r"you\s+always\s+do\s+this",
            r"no\s+one\s+will\s+believe\s+you",
            r"you'?re\s+too\s+sensitive",
            r"jy\s+is\s+(mal|gek)",
            r"dit\s+het\s+nooit\s+gebeur",
            r"jy\s+verbeel\s+jou",
            r"ek\s+het\s+dit\s+nooit\s+gesê",
        ]

        gaslighting_instances = []
        
        for msg in messages:
            text_lower = msg.text.lower()
            for pattern in gaslighting_phrases:
                if re.search(pattern, text_lower):
                    gaslighting_instances.append({
                        'message': msg,
                        'pattern': pattern
                    })

        if gaslighting_instances:
            score = len(gaslighting_instances) / len(messages)
            
            if score >= GASLIGHTING_THRESHOLD:
                self.finding_counter += 1
                findings.append(ForensicFinding(
                    finding_id=f"PS_{self.finding_counter:04d}",
                    analysis_type=AnalysisType.PSYCHOLOGICAL,
                    severity=SeverityLevel.CRITICAL if score > 0.1 else SeverityLevel.HIGH,
                    title="Gaslighting Pattern Detected",
                    description=f"Found {len(gaslighting_instances)} instances of potential gaslighting language.",
                    evidence=[
                        f"{g['message'].sender}: {g['message'].text[:100]}"
                        for g in gaslighting_instances[:5]
                    ],
                    message_ids=[g['message'].message_id for g in gaslighting_instances],
                    confidence=min(0.95, 0.6 + score * 2)
                ))

        return findings

    def _detect_manipulation(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Detect manipulation patterns"""
        findings = []
        
        manipulation_patterns = {
            'guilt_tripping': [
                r"after\s+all\s+i('ve)?\s+(done|did)",
                r"you\s+don'?t\s+(love|care)",
                r"if\s+you\s+(loved|cared)",
                r"na\s+alles\s+wat\s+ek",
                r"jy\s+gee\s+nie\s+om",
            ],
            'threatening': [
                r"if\s+you\s+leave",
                r"you'?ll\s+(regret|be\s+sorry)",
                r"i'?ll\s+tell\s+everyone",
                r"as\s+jy\s+loop",
                r"jy\s+sal\s+spyt\s+wees",
            ],
            'love_bombing': [
                r"you'?re\s+the\s+(only|best)",
                r"no\s+one\s+(understands|loves)\s+you\s+like\s+i",
                r"we'?re\s+meant\s+to\s+be",
                r"can'?t\s+live\s+without\s+you",
            ],
            'isolation': [
                r"(your|those)\s+friends\s+(don't|are)",
                r"your\s+family\s+(doesn't|hates)",
                r"they'?re\s+trying\s+to",
                r"only\s+i\s+(understand|love|care)",
            ]
        }

        manipulation_found = {}
        
        for msg in messages:
            text_lower = msg.text.lower()
            for manip_type, patterns in manipulation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        if manip_type not in manipulation_found:
                            manipulation_found[manip_type] = []
                        manipulation_found[manip_type].append(msg)
                        break

        for manip_type, instances in manipulation_found.items():
            if len(instances) >= 2:
                self.finding_counter += 1
                findings.append(ForensicFinding(
                    finding_id=f"PS_{self.finding_counter:04d}",
                    analysis_type=AnalysisType.PSYCHOLOGICAL,
                    severity=SeverityLevel.HIGH,
                    title=f"Manipulation Pattern: {manip_type.replace('_', ' ').title()}",
                    description=f"Found {len(instances)} instances of {manip_type} behavior.",
                    evidence=[
                        f"{m.sender}: {m.text[:80]}..."
                        for m in instances[:3]
                    ],
                    message_ids=[m.message_id for m in instances],
                    confidence=0.80
                ))

        return findings

    def _analyze_emotional_patterns(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """Analyze emotional patterns over time"""
        findings = []
        
        # Track emotional indicators
        emotions = {
            'anger': [r'angry', r'furious', r'hate', r'kwaad', r'woedend'],
            'fear': [r'afraid', r'scared', r'worried', r'bang', r'bekommerd'],
            'sadness': [r'sad', r'depressed', r'crying', r'hartseer', r'huil'],
            'anxiety': [r'anxious', r'stressed', r'panic', r'angstig'],
        }

        emotion_timeline = []
        
        for msg in messages:
            text_lower = msg.text.lower()
            msg_emotions = []
            
            for emotion, patterns in emotions.items():
                if any(re.search(p, text_lower) for p in patterns):
                    msg_emotions.append(emotion)
            
            if msg_emotions:
                emotion_timeline.append({
                    'message': msg,
                    'emotions': msg_emotions
                })

        # Look for emotional escalation
        if len(emotion_timeline) >= 3:
            anger_count = sum(1 for e in emotion_timeline if 'anger' in e['emotions'])
            fear_count = sum(1 for e in emotion_timeline if 'fear' in e['emotions'])
            
            if anger_count >= 3 or fear_count >= 3:
                self.finding_counter += 1
                findings.append(ForensicFinding(
                    finding_id=f"PS_{self.finding_counter:04d}",
                    analysis_type=AnalysisType.PSYCHOLOGICAL,
                    severity=SeverityLevel.MEDIUM,
                    title="Emotional Pattern Detected",
                    description=f"Recurring emotional indicators: anger={anger_count}, fear={fear_count}",
                    evidence=[
                        f"{e['message'].sender}: {', '.join(e['emotions'])}"
                        for e in emotion_timeline[:5]
                    ],
                    message_ids=[e['message'].message_id for e in emotion_timeline],
                    confidence=0.70
                ))

        return findings

    def _ai_psychological_analysis(
        self,
        messages: List[TextMessage]
    ) -> List[ForensicFinding]:
        """AI-powered psychological analysis"""
        findings = []
        
        try:
            # Prepare conversation
            conversation = "\n".join([
                f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.text}"
                for m in sorted(messages, key=lambda x: x.timestamp)[:30]
            ])

            prompt = f"""Perform psychological analysis of this conversation.

CONVERSATION:
{conversation}

ANALYZE FOR:
1. Power dynamics between speakers
2. Signs of emotional manipulation
3. Gaslighting indicators
4. Narcissistic behavior patterns
5. Anxiety or fear indicators
6. Controlling behavior
7. Trust issues
8. Communication health

RESPOND IN JSON:
{{
    "power_dynamics": {{
        "dominant_speaker": "name or null",
        "imbalance_score": 0-100,
        "evidence": ["quotes"]
    }},
    "manipulation_indicators": [
        {{
            "type": "type",
            "speaker": "name",
            "evidence": "quote",
            "severity": "high/medium/low"
        }}
    ],
    "gaslighting_score": 0-100,
    "narcissistic_indicators": 0-100,
    "relationship_health": 0-100,
    "concerns": ["list of psychological concerns"],
    "overall_assessment": "summary"
}}"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )

            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Create findings from analysis
                if result.get('gaslighting_score', 0) > 50:
                    self.finding_counter += 1
                    findings.append(ForensicFinding(
                        finding_id=f"PS_{self.finding_counter:04d}",
                        analysis_type=AnalysisType.PSYCHOLOGICAL,
                        severity=SeverityLevel.CRITICAL if result['gaslighting_score'] > 70 else SeverityLevel.HIGH,
                        title="Gaslighting Detected (AI Analysis)",
                        description=f"AI detected gaslighting behavior (score: {result['gaslighting_score']}/100)",
                        evidence=result.get('concerns', []),
                        message_ids=[],
                        confidence=result['gaslighting_score'] / 100
                    ))

                for manip in result.get('manipulation_indicators', []):
                    self.finding_counter += 1
                    findings.append(ForensicFinding(
                        finding_id=f"PS_{self.finding_counter:04d}",
                        analysis_type=AnalysisType.PSYCHOLOGICAL,
                        severity=SeverityLevel.HIGH if manip.get('severity') == 'high' else SeverityLevel.MEDIUM,
                        title=f"Manipulation: {manip.get('type', 'Unknown')}",
                        description=f"Speaker {manip.get('speaker', 'Unknown')} showing manipulation behavior",
                        evidence=[manip.get('evidence', '')],
                        message_ids=[],
                        confidence=0.80
                    ))

        except Exception as e:
            logger.error(f"AI psychological analysis error: {e}")

        return findings


class TextForensicsEngine:
    """
    Main text forensics engine
    Combines all analysis types
    """

    def __init__(self):
        """Initialize text forensics engine"""
        self.pattern_detector = PatternChangeDetector()
        self.story_analyzer = StoryFlowAnalyzer()
        self.contradiction_detector = ContradictionDetector()
        self.psychological_analyzer = PsychologicalAnalyzer()
        self.report_counter = 0

    def analyze_conversation(
        self,
        messages: List[TextMessage],
        topic: str = ""
    ) -> ForensicReport:
        """
        Perform complete forensic analysis on conversation

        Args:
            messages: List of messages to analyze
            topic: Topic of conversation (optional)

        Returns:
            Complete forensic report
        """
        self.report_counter += 1
        report_id = f"TFR_{self.report_counter:06d}"
        
        logger.info(f"Starting text forensic analysis: {report_id}")
        
        all_findings = []

        # Get unique senders
        senders = list(set(m.sender for m in messages))

        # Pattern change detection for each sender
        for sender in senders:
            pattern_findings = self.pattern_detector.analyze_writing_patterns(messages, sender)
            all_findings.extend(pattern_findings)

        # Story flow analysis
        story_findings = self.story_analyzer.analyze_story_flow(messages, topic)
        all_findings.extend(story_findings)

        # Contradiction detection
        contradiction_findings = self.contradiction_detector.detect_contradictions(messages)
        all_findings.extend(contradiction_findings)

        # Psychological analysis
        psych_findings = self.psychological_analyzer.analyze_psychological_patterns(messages)
        all_findings.extend(psych_findings)

        # Sort findings by severity
        severity_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3,
            SeverityLevel.INFO: 4
        }
        all_findings.sort(key=lambda x: severity_order.get(x.severity, 5))

        # Generate summary
        summary = self._generate_summary(messages, senders, all_findings)
        
        # Risk assessment
        risk = self._assess_risk(all_findings)

        # Date range
        timestamps = [m.timestamp for m in messages]
        date_range = {
            'start': min(timestamps).isoformat(),
            'end': max(timestamps).isoformat()
        }

        return ForensicReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            message_count=len(messages),
            sender_count=len(senders),
            date_range=date_range,
            findings=all_findings,
            summary=summary,
            risk_assessment=risk
        )

    def _generate_summary(
        self,
        messages: List[TextMessage],
        senders: List[str],
        findings: List[ForensicFinding]
    ) -> Dict[str, Any]:
        """Generate analysis summary"""
        by_type = {}
        by_severity = {}
        
        for f in findings:
            # By type
            atype = f.analysis_type.value
            by_type[atype] = by_type.get(atype, 0) + 1
            
            # By severity
            sev = f.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            'total_findings': len(findings),
            'by_analysis_type': by_type,
            'by_severity': by_severity,
            'senders_analyzed': senders,
            'message_count': len(messages),
            'critical_count': by_severity.get('critical', 0),
            'high_count': by_severity.get('high', 0)
        }

    def _assess_risk(self, findings: List[ForensicFinding]) -> Dict[str, Any]:
        """Assess overall risk level"""
        critical = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        medium = sum(1 for f in findings if f.severity == SeverityLevel.MEDIUM)
        
        # Calculate risk score
        risk_score = (critical * 30) + (high * 15) + (medium * 5)
        risk_score = min(100, risk_score)
        
        if risk_score >= 70:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        elif risk_score >= 10:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'critical_findings': critical,
            'high_findings': high,
            'recommendations': self._get_recommendations(risk_level, findings)
        }

    def _get_recommendations(
        self,
        risk_level: str,
        findings: List[ForensicFinding]
    ) -> List[str]:
        """Get recommendations based on findings"""
        recommendations = []
        
        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.append("Immediate review of flagged communications recommended")
        
        # Check for specific patterns
        has_gaslighting = any(
            'gaslighting' in f.title.lower()
            for f in findings
        )
        has_contradictions = any(
            f.analysis_type == AnalysisType.CONTRADICTION
            for f in findings
        )
        has_manipulation = any(
            'manipulation' in f.title.lower()
            for f in findings
        )

        if has_gaslighting:
            recommendations.append("Evidence of gaslighting behavior - document all instances")
        
        if has_contradictions:
            recommendations.append("Multiple contradictions found - create timeline to verify facts")
        
        if has_manipulation:
            recommendations.append("Manipulation patterns detected - consider professional consultation")

        if not recommendations:
            recommendations.append("Continue monitoring - no immediate concerns")

        return recommendations


if __name__ == "__main__":
    engine = TextForensicsEngine()
    print("Text Forensics Engine initialized")
    print("Use engine.analyze_conversation(messages) for full analysis")
