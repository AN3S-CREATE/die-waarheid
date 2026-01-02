"""
Expert Panel Commentary System for Die Waarheid
Forensic experts analyze each evidence item with contextual commentary
Cross-references with past evidence to identify contradictions and patterns

EXPERT ROLES:
1. Linguistic Expert - Language patterns, authenticity, Afrikaans nuances
2. Psychological Expert - Behavior patterns, manipulation, stress indicators
3. Forensic Expert - Timeline consistency, evidence validity, contradictions
4. Audio Expert - Voice analysis, stress levels, authenticity
5. Investigative Expert - Pattern recognition, case strategy, risk assessment
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

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
    GEMINI_SAFETY_SETTINGS
)

logger = logging.getLogger(__name__)


class ExpertRole(Enum):
    """Expert panel roles"""
    LINGUISTIC = "linguistic_expert"
    PSYCHOLOGICAL = "psychological_expert"
    FORENSIC = "forensic_expert"
    AUDIO = "audio_expert"
    INVESTIGATIVE = "investigative_expert"


class CommentarySeverity(Enum):
    """Severity of expert commentary"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class ExpertComment:
    """Single expert commentary"""
    comment_id: str
    expert_role: ExpertRole
    evidence_id: str
    timestamp: str
    
    # Commentary
    title: str
    analysis: str
    key_findings: List[str]
    severity: CommentarySeverity
    
    # Cross-references
    references_past_evidence: List[str]
    contradictions: List[Dict[str, str]]
    pattern_observations: List[str]
    
    # Recommendations
    recommendations: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'comment_id': self.comment_id,
            'expert_role': self.expert_role.value,
            'evidence_id': self.evidence_id,
            'timestamp': self.timestamp,
            'title': self.title,
            'analysis': self.analysis,
            'key_findings': self.key_findings,
            'severity': self.severity.value,
            'references_past_evidence': self.references_past_evidence,
            'contradictions': self.contradictions,
            'pattern_observations': self.pattern_observations,
            'recommendations': self.recommendations,
            'confidence': self.confidence
        }


@dataclass
class ExpertBrief:
    """Complete expert brief for an evidence item"""
    brief_id: str
    evidence_id: str
    generated_at: str
    
    # Comments from all experts
    expert_comments: List[ExpertComment]
    
    # Synthesis
    overall_assessment: str
    critical_issues: List[str]
    pattern_changes: List[str]
    timeline_concerns: List[str]
    
    # Risk assessment
    risk_score: float
    risk_level: str
    
    # Next steps
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'brief_id': self.brief_id,
            'evidence_id': self.evidence_id,
            'generated_at': self.generated_at,
            'expert_comments': [c.to_dict() for c in self.expert_comments],
            'overall_assessment': self.overall_assessment,
            'critical_issues': self.critical_issues,
            'pattern_changes': self.pattern_changes,
            'timeline_concerns': self.timeline_concerns,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'recommended_actions': self.recommended_actions
        }


class LinguisticExpert:
    """Linguistic analysis expert"""
    
    def __init__(self):
        """Initialize linguistic expert"""
        self.comment_counter = 0
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

    def analyze_evidence(
        self,
        evidence_text: str,
        evidence_id: str,
        sender: str,
        past_evidence: List[Dict[str, Any]]
    ) -> Optional[ExpertComment]:
        """
        Analyze evidence from linguistic perspective

        Args:
            evidence_text: Text to analyze
            evidence_id: Evidence identifier
            sender: Speaker/sender
            past_evidence: Previous evidence from same sender

        Returns:
            Expert comment
        """
        if not self.gemini_configured or not evidence_text:
            return None

        try:
            self.comment_counter += 1
            comment_id = f"LC_{self.comment_counter:06d}"

            # Prepare past evidence summary
            past_summary = self._summarize_past_evidence(past_evidence, sender)

            prompt = f"""You are a forensic linguistic expert analyzing communication for an investigation.

CURRENT EVIDENCE:
Text: "{evidence_text}"
Speaker: {sender}
Evidence ID: {evidence_id}

PAST EVIDENCE FROM SAME SPEAKER:
{past_summary}

ANALYZE FOR:
1. Language patterns and vocabulary consistency
2. Grammatical structures and writing style
3. Afrikaans language authenticity and nuances
4. Emotional tone and intensity changes
5. Consistency with past communications
6. Any signs of deception or fabrication
7. Linguistic indicators of stress or manipulation

SPECIFICALLY LOOK FOR:
- Vocabulary shifts (new words, dropped words)
- Style changes (formal to casual or vice versa)
- Grammatical inconsistencies
- Afrikaans authenticity issues
- Tone changes compared to past
- Defensive language patterns
- Contradictions in stated facts

RESPOND IN JSON:
{{
    "title": "Brief title of finding",
    "analysis": "Detailed linguistic analysis",
    "key_findings": ["finding1", "finding2", "finding3"],
    "severity": "critical/high/medium/low/informational",
    "references_past_evidence": ["evidence_id1", "evidence_id2"],
    "contradictions": [
        {{
            "current_statement": "what they say now",
            "past_statement": "what they said before",
            "implication": "what this means"
        }}
    ],
    "pattern_observations": [
        "Pattern observation 1",
        "Pattern observation 2"
    ],
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence": 0.85
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
                
                severity_map = {
                    'critical': CommentarySeverity.CRITICAL,
                    'high': CommentarySeverity.HIGH,
                    'medium': CommentarySeverity.MEDIUM,
                    'low': CommentarySeverity.LOW,
                    'informational': CommentarySeverity.INFORMATIONAL
                }

                return ExpertComment(
                    comment_id=comment_id,
                    expert_role=ExpertRole.LINGUISTIC,
                    evidence_id=evidence_id,
                    timestamp=datetime.now().isoformat(),
                    title=result.get('title', 'Linguistic Analysis'),
                    analysis=result.get('analysis', ''),
                    key_findings=result.get('key_findings', []),
                    severity=severity_map.get(result.get('severity', 'medium'), CommentarySeverity.MEDIUM),
                    references_past_evidence=result.get('references_past_evidence', []),
                    contradictions=result.get('contradictions', []),
                    pattern_observations=result.get('pattern_observations', []),
                    recommendations=result.get('recommendations', []),
                    confidence=result.get('confidence', 0.7)
                )

        except Exception as e:
            logger.error(f"Linguistic analysis error: {e}")
            return None

    def _summarize_past_evidence(
        self,
        past_evidence: List[Dict[str, Any]],
        sender: str
    ) -> str:
        """Summarize past evidence from same sender"""
        if not past_evidence:
            return "No past evidence from this speaker."

        summary = []
        for ev in past_evidence[-5:]:  # Last 5 items
            timestamp = ev.get('timestamp', 'Unknown')
            text = ev.get('text', '')[:200]
            summary.append(f"[{timestamp}]: {text}...")

        return "\n".join(summary)


class PsychologicalExpert:
    """Psychological analysis expert"""
    
    def __init__(self):
        """Initialize psychological expert"""
        self.comment_counter = 0
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

    def analyze_evidence(
        self,
        evidence_text: str,
        evidence_id: str,
        sender: str,
        past_evidence: List[Dict[str, Any]],
        audio_stress: Optional[float] = None
    ) -> Optional[ExpertComment]:
        """
        Analyze evidence from psychological perspective

        Args:
            evidence_text: Text to analyze
            evidence_id: Evidence identifier
            sender: Speaker/sender
            past_evidence: Previous evidence from same sender
            audio_stress: Stress level if audio (0-100)

        Returns:
            Expert comment
        """
        if not self.gemini_configured or not evidence_text:
            return None

        try:
            self.comment_counter += 1
            comment_id = f"PC_{self.comment_counter:06d}"

            # Prepare context
            past_summary = self._summarize_past_evidence(past_evidence)
            stress_context = f"Audio stress level: {audio_stress}/100" if audio_stress else "Text-based evidence"

            prompt = f"""You are a forensic psychologist analyzing communication for an investigation.

CURRENT EVIDENCE:
Text: "{evidence_text}"
Speaker: {sender}
Evidence ID: {evidence_id}
Context: {stress_context}

PAST EVIDENCE FROM SAME SPEAKER:
{past_summary}

PSYCHOLOGICAL ANALYSIS:
1. Emotional state and changes
2. Stress indicators
3. Manipulation tactics (gaslighting, guilt-tripping, isolation)
4. Defensiveness and denial patterns
5. Narcissistic indicators
6. Victim mentality vs. accountability
7. Changes in emotional presentation over time
8. Consistency of emotional responses

CRITICAL QUESTIONS:
- Is the emotional response consistent with past behavior?
- Are there signs of manipulation or control?
- Does the speaker take responsibility or blame others?
- Are there contradictions between stated emotions and actions?
- What psychological pattern is emerging?

RESPOND IN JSON:
{{
    "title": "Psychological assessment title",
    "analysis": "Detailed psychological analysis",
    "key_findings": ["finding1", "finding2"],
    "severity": "critical/high/medium/low/informational",
    "references_past_evidence": ["evidence_id1"],
    "contradictions": [
        {{
            "current_behavior": "what they do now",
            "past_behavior": "what they did before",
            "implication": "psychological implication"
        }}
    ],
    "pattern_observations": [
        "Pattern 1: Description",
        "Pattern 2: Description"
    ],
    "recommendations": ["recommendation1"],
    "confidence": 0.80
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
                
                severity_map = {
                    'critical': CommentarySeverity.CRITICAL,
                    'high': CommentarySeverity.HIGH,
                    'medium': CommentarySeverity.MEDIUM,
                    'low': CommentarySeverity.LOW,
                    'informational': CommentarySeverity.INFORMATIONAL
                }

                return ExpertComment(
                    comment_id=comment_id,
                    expert_role=ExpertRole.PSYCHOLOGICAL,
                    evidence_id=evidence_id,
                    timestamp=datetime.now().isoformat(),
                    title=result.get('title', 'Psychological Analysis'),
                    analysis=result.get('analysis', ''),
                    key_findings=result.get('key_findings', []),
                    severity=severity_map.get(result.get('severity', 'medium'), CommentarySeverity.MEDIUM),
                    references_past_evidence=result.get('references_past_evidence', []),
                    contradictions=result.get('contradictions', []),
                    pattern_observations=result.get('pattern_observations', []),
                    recommendations=result.get('recommendations', []),
                    confidence=result.get('confidence', 0.7)
                )

        except Exception as e:
            logger.error(f"Psychological analysis error: {e}")
            return None

    def _summarize_past_evidence(self, past_evidence: List[Dict[str, Any]]) -> str:
        """Summarize past evidence"""
        if not past_evidence:
            return "No past evidence available."

        summary = []
        for ev in past_evidence[-5:]:
            timestamp = ev.get('timestamp', 'Unknown')
            text = ev.get('text', '')[:150]
            summary.append(f"[{timestamp}]: {text}...")

        return "\n".join(summary)


class ForensicExpert:
    """Forensic analysis expert"""
    
    def __init__(self):
        """Initialize forensic expert"""
        self.comment_counter = 0
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

    def analyze_evidence(
        self,
        evidence_text: str,
        evidence_id: str,
        sender: str,
        evidence_timestamp: datetime,
        past_evidence: List[Dict[str, Any]]
    ) -> Optional[ExpertComment]:
        """
        Analyze evidence from forensic perspective

        Args:
            evidence_text: Text to analyze
            evidence_id: Evidence identifier
            sender: Speaker/sender
            evidence_timestamp: When evidence was created
            past_evidence: Previous evidence

        Returns:
            Expert comment
        """
        if not self.gemini_configured or not evidence_text:
            return None

        try:
            self.comment_counter += 1
            comment_id = f"FE_{self.comment_counter:06d}"

            # Prepare timeline context
            timeline_context = self._build_timeline_context(past_evidence, evidence_timestamp)

            prompt = f"""You are a forensic investigator analyzing evidence for an investigation.

CURRENT EVIDENCE:
Text: "{evidence_text}"
Speaker: {sender}
Evidence ID: {evidence_id}
Timestamp: {evidence_timestamp.isoformat()}

TIMELINE CONTEXT:
{timeline_context}

FORENSIC ANALYSIS:
1. Timeline consistency - does the story match the timeline?
2. Factual contradictions - are facts consistent?
3. Sequence of events - does the sequence make sense?
4. Alibi consistency - can the story be verified?
5. Evidence validity - is this evidence authentic?
6. Gaps in narrative - what's missing?
7. Implausibilities - what doesn't add up?

CRITICAL QUESTIONS:
- Why is the speaker saying this NOW?
- How does this fit with what happened BEFORE?
- Are there timeline inconsistencies?
- Do the facts align with other evidence?
- What is the speaker NOT saying?

RESPOND IN JSON:
{{
    "title": "Forensic assessment title",
    "analysis": "Detailed forensic analysis",
    "key_findings": ["finding1", "finding2"],
    "severity": "critical/high/medium/low/informational",
    "references_past_evidence": ["evidence_id1"],
    "contradictions": [
        {{
            "current_claim": "what they claim now",
            "conflicting_evidence": "what contradicts it",
            "implication": "forensic implication"
        }}
    ],
    "pattern_observations": [
        "Timeline observation 1",
        "Factual observation 2"
    ],
    "recommendations": ["recommendation1"],
    "confidence": 0.85
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
                
                severity_map = {
                    'critical': CommentarySeverity.CRITICAL,
                    'high': CommentarySeverity.HIGH,
                    'medium': CommentarySeverity.MEDIUM,
                    'low': CommentarySeverity.LOW,
                    'informational': CommentarySeverity.INFORMATIONAL
                }

                return ExpertComment(
                    comment_id=comment_id,
                    expert_role=ExpertRole.FORENSIC,
                    evidence_id=evidence_id,
                    timestamp=datetime.now().isoformat(),
                    title=result.get('title', 'Forensic Analysis'),
                    analysis=result.get('analysis', ''),
                    key_findings=result.get('key_findings', []),
                    severity=severity_map.get(result.get('severity', 'medium'), CommentarySeverity.MEDIUM),
                    references_past_evidence=result.get('references_past_evidence', []),
                    contradictions=result.get('contradictions', []),
                    pattern_observations=result.get('pattern_observations', []),
                    recommendations=result.get('recommendations', []),
                    confidence=result.get('confidence', 0.7)
                )

        except Exception as e:
            logger.error(f"Forensic analysis error: {e}")
            return None

    def _build_timeline_context(
        self,
        past_evidence: List[Dict[str, Any]],
        current_timestamp: datetime
    ) -> str:
        """Build timeline context"""
        if not past_evidence:
            return "No past evidence to compare."

        context = []
        for ev in past_evidence[-10:]:
            ts = ev.get('timestamp')
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                time_diff = (current_timestamp - ts).days
                text = ev.get('text', '')[:100]
                context.append(f"[{time_diff} days ago]: {text}...")

        return "\n".join(context)


class AudioExpert:
    """Audio analysis expert"""
    
    def __init__(self):
        """Initialize audio expert"""
        self.comment_counter = 0

    def analyze_evidence(
        self,
        evidence_id: str,
        sender: str,
        audio_metrics: Dict[str, float],
        past_audio: List[Dict[str, Any]]
    ) -> Optional[ExpertComment]:
        """
        Analyze audio evidence

        Args:
            evidence_id: Evidence identifier
            sender: Speaker
            audio_metrics: Audio analysis metrics (stress, pitch, etc.)
            past_audio: Past audio evidence from same speaker

        Returns:
            Expert comment
        """
        try:
            self.comment_counter += 1
            comment_id = f"AE_{self.comment_counter:06d}"

            # Analyze metrics
            stress_level = audio_metrics.get('stress_level', 0)
            pitch_volatility = audio_metrics.get('pitch_volatility', 0)
            silence_ratio = audio_metrics.get('silence_ratio', 0)

            # Compare with past
            past_stress = [a.get('stress_level', 0) for a in past_audio]
            avg_past_stress = sum(past_stress) / len(past_stress) if past_stress else 0

            key_findings = []
            contradictions = []
            pattern_observations = []
            recommendations = []
            severity = CommentarySeverity.INFORMATIONAL

            # Analyze stress level
            if stress_level > 70:
                key_findings.append(f"High stress level detected: {stress_level:.0f}/100")
                severity = CommentarySeverity.HIGH
                recommendations.append("Investigate cause of elevated stress")
            elif stress_level > 50:
                key_findings.append(f"Moderate stress level: {stress_level:.0f}/100")
                severity = CommentarySeverity.MEDIUM

            # Compare with past
            if past_stress and stress_level > avg_past_stress * 1.5:
                pattern_observations.append(
                    f"Stress level significantly higher than average ({avg_past_stress:.0f})"
                )
                contradictions.append({
                    'current_behavior': f'High stress ({stress_level:.0f})',
                    'past_behavior': f'Average stress ({avg_past_stress:.0f})',
                    'implication': 'Speaker may be under unusual pressure or stress'
                })

            # Pitch volatility
            if pitch_volatility > 0.7:
                key_findings.append("High pitch volatility - emotional instability")
                pattern_observations.append("Voice shows signs of emotional distress")

            # Silence patterns
            if silence_ratio > 0.4:
                key_findings.append("High silence ratio - hesitation or thinking time")
                pattern_observations.append("Speaker pauses frequently - possible deception or careful speech")

            return ExpertComment(
                comment_id=comment_id,
                expert_role=ExpertRole.AUDIO,
                evidence_id=evidence_id,
                timestamp=datetime.now().isoformat(),
                title=f"Audio Analysis - {sender}",
                analysis=f"Stress: {stress_level:.0f}, Pitch Volatility: {pitch_volatility:.2f}, Silence: {silence_ratio:.1%}",
                key_findings=key_findings,
                severity=severity,
                references_past_evidence=[],
                contradictions=contradictions,
                pattern_observations=pattern_observations,
                recommendations=recommendations,
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return None


class InvestigativeExpert:
    """Investigative strategy expert"""
    
    def __init__(self):
        """Initialize investigative expert"""
        self.comment_counter = 0
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

    def analyze_evidence(
        self,
        evidence_id: str,
        all_comments: List[ExpertComment],
        all_evidence: List[Dict[str, Any]]
    ) -> Optional[ExpertComment]:
        """
        Synthesize all expert opinions into investigative strategy

        Args:
            evidence_id: Evidence identifier
            all_comments: All expert comments for this evidence
            all_evidence: All evidence in case

        Returns:
            Investigative expert comment
        """
        if not self.gemini_configured or not all_comments:
            return None

        try:
            self.comment_counter += 1
            comment_id = f"IE_{self.comment_counter:06d}"

            # Synthesize comments
            synthesis = self._synthesize_comments(all_comments)

            prompt = f"""You are an investigative strategy expert synthesizing expert opinions.

EXPERT OPINIONS SUMMARY:
{synthesis}

TOTAL EVIDENCE IN CASE: {len(all_evidence)} items

INVESTIGATIVE STRATEGY:
1. What is the most important finding?
2. What patterns are emerging?
3. What should be investigated next?
4. What evidence is most critical?
5. What are the key contradictions?
6. What is the risk level?
7. What are the next investigative steps?

RESPOND IN JSON:
{{
    "title": "Investigative assessment",
    "analysis": "Strategic analysis and recommendations",
    "key_findings": ["finding1", "finding2"],
    "severity": "critical/high/medium/low",
    "pattern_observations": [
        "Pattern 1",
        "Pattern 2"
    ],
    "recommendations": [
        "Next investigative step 1",
        "Next investigative step 2"
    ],
    "confidence": 0.80
}}"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1500
                )
            )

            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                result = json.loads(json_match.group())
                
                severity_map = {
                    'critical': CommentarySeverity.CRITICAL,
                    'high': CommentarySeverity.HIGH,
                    'medium': CommentarySeverity.MEDIUM,
                    'low': CommentarySeverity.LOW
                }

                return ExpertComment(
                    comment_id=comment_id,
                    expert_role=ExpertRole.INVESTIGATIVE,
                    evidence_id=evidence_id,
                    timestamp=datetime.now().isoformat(),
                    title=result.get('title', 'Investigative Assessment'),
                    analysis=result.get('analysis', ''),
                    key_findings=result.get('key_findings', []),
                    severity=severity_map.get(result.get('severity', 'medium'), CommentarySeverity.MEDIUM),
                    references_past_evidence=[],
                    contradictions=[],
                    pattern_observations=result.get('pattern_observations', []),
                    recommendations=result.get('recommendations', []),
                    confidence=result.get('confidence', 0.7)
                )

        except Exception as e:
            logger.error(f"Investigative analysis error: {e}")
            return None

    def _synthesize_comments(self, comments: List[ExpertComment]) -> str:
        """Synthesize expert comments"""
        synthesis = []
        for comment in comments:
            synthesis.append(f"\n{comment.expert_role.value.upper()}:")
            synthesis.append(f"  Title: {comment.title}")
            synthesis.append(f"  Severity: {comment.severity.value}")
            synthesis.append(f"  Key Findings: {', '.join(comment.key_findings)}")
            if comment.contradictions:
                synthesis.append(f"  Contradictions: {len(comment.contradictions)}")

        return "\n".join(synthesis)


class ExpertPanelAnalyzer:
    """
    Main expert panel analyzer
    Coordinates all experts for comprehensive analysis
    """

    def __init__(self):
        """Initialize expert panel"""
        self.linguistic_expert = LinguisticExpert()
        self.psychological_expert = PsychologicalExpert()
        self.forensic_expert = ForensicExpert()
        self.audio_expert = AudioExpert()
        self.investigative_expert = InvestigativeExpert()
        self.brief_counter = 0

    def generate_expert_brief(
        self,
        evidence_id: str,
        evidence_text: str,
        evidence_type: str,
        sender: str,
        evidence_timestamp: datetime,
        audio_metrics: Optional[Dict[str, float]] = None,
        past_evidence: Optional[List[Dict[str, Any]]] = None
    ) -> ExpertBrief:
        """
        Generate comprehensive expert brief for evidence

        Args:
            evidence_id: Evidence identifier
            evidence_text: Text content
            evidence_type: Type of evidence
            sender: Speaker/sender
            evidence_timestamp: When evidence was created
            audio_metrics: Audio analysis metrics (if applicable)
            past_evidence: Previous evidence from same sender

        Returns:
            Complete expert brief
        """
        self.brief_counter += 1
        brief_id = f"EB_{self.brief_counter:06d}"

        if past_evidence is None:
            past_evidence = []

        logger.info(f"Generating expert brief for {evidence_id}")

        # Collect all expert comments
        expert_comments = []

        # Linguistic analysis
        if evidence_text:
            ling_comment = self.linguistic_expert.analyze_evidence(
                evidence_text, evidence_id, sender, past_evidence
            )
            if ling_comment:
                expert_comments.append(ling_comment)

        # Psychological analysis
        if evidence_text:
            psych_comment = self.psychological_expert.analyze_evidence(
                evidence_text, evidence_id, sender, past_evidence,
                audio_stress=audio_metrics.get('stress_level') if audio_metrics else None
            )
            if psych_comment:
                expert_comments.append(psych_comment)

        # Forensic analysis
        if evidence_text:
            forensic_comment = self.forensic_expert.analyze_evidence(
                evidence_text, evidence_id, sender, evidence_timestamp, past_evidence
            )
            if forensic_comment:
                expert_comments.append(forensic_comment)

        # Audio analysis
        if audio_metrics:
            audio_comment = self.audio_expert.analyze_evidence(
                evidence_id, sender, audio_metrics,
                [e for e in past_evidence if e.get('audio_metrics')]
            )
            if audio_comment:
                expert_comments.append(audio_comment)

        # Investigative synthesis
        inv_comment = self.investigative_expert.analyze_evidence(
            evidence_id, expert_comments, past_evidence
        )
        if inv_comment:
            expert_comments.append(inv_comment)

        # Synthesize brief
        brief = self._synthesize_brief(brief_id, evidence_id, expert_comments)

        return brief

    def _synthesize_brief(
        self,
        brief_id: str,
        evidence_id: str,
        expert_comments: List[ExpertComment]
    ) -> ExpertBrief:
        """Synthesize expert comments into brief"""
        
        # Extract critical issues
        critical_issues = [
            c.title for c in expert_comments
            if c.severity == CommentarySeverity.CRITICAL
        ]

        # Extract pattern changes
        pattern_changes = []
        for comment in expert_comments:
            pattern_changes.extend(comment.pattern_observations)

        # Extract timeline concerns
        timeline_concerns = []
        for comment in expert_comments:
            if comment.expert_role == ExpertRole.FORENSIC:
                timeline_concerns.extend(comment.key_findings)

        # Calculate risk score
        severity_weights = {
            CommentarySeverity.CRITICAL: 30,
            CommentarySeverity.HIGH: 20,
            CommentarySeverity.MEDIUM: 10,
            CommentarySeverity.LOW: 5,
            CommentarySeverity.INFORMATIONAL: 1
        }

        risk_score = sum(
            severity_weights.get(c.severity, 0) * c.confidence
            for c in expert_comments
        ) / max(len(expert_comments), 1)

        risk_score = min(100, risk_score)

        # Determine risk level
        if risk_score >= 70:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Collect recommendations
        recommendations = []
        for comment in expert_comments:
            recommendations.extend(comment.recommendations)

        # Remove duplicates
        recommendations = list(set(recommendations))

        # Overall assessment
        overall_assessment = self._generate_overall_assessment(expert_comments)

        return ExpertBrief(
            brief_id=brief_id,
            evidence_id=evidence_id,
            generated_at=datetime.now().isoformat(),
            expert_comments=expert_comments,
            overall_assessment=overall_assessment,
            critical_issues=critical_issues,
            pattern_changes=pattern_changes,
            timeline_concerns=timeline_concerns,
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_actions=recommendations
        )

    def _generate_overall_assessment(self, comments: List[ExpertComment]) -> str:
        """Generate overall assessment"""
        if not comments:
            return "No expert analysis available."

        # Find most severe comment
        most_severe = max(comments, key=lambda c: list(CommentarySeverity).index(c.severity))

        assessment = f"Expert panel analysis reveals {most_severe.title.lower()}. "

        # Count findings
        total_findings = sum(len(c.key_findings) for c in comments)
        assessment += f"Total findings: {total_findings}. "

        # Contradictions
        total_contradictions = sum(len(c.contradictions) for c in comments)
        if total_contradictions > 0:
            assessment += f"Contradictions identified: {total_contradictions}. "

        # Patterns
        total_patterns = sum(len(c.pattern_observations) for c in comments)
        if total_patterns > 0:
            assessment += f"Pattern observations: {total_patterns}."

        return assessment


if __name__ == "__main__":
    panel = ExpertPanelAnalyzer()
    print("Expert Panel Analyzer initialized")
    print("Use panel.generate_expert_brief() to analyze evidence")
