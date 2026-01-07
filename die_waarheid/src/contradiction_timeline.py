"""
Contradiction Timeline Visualization System for Die Waarheid
Creates interactive timeline showing statement contradictions
Highlights when contradictions occur and what changed between statements

FEATURES:
- Visual timeline of all statements
- Contradiction highlighting
- Time gaps between statements
- Pattern visualization
- Interactive HTML export
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class StatementType(Enum):
    """Type of statement"""
    TEXT_MESSAGE = "text_message"
    VOICE_NOTE = "voice_note"
    TRANSCRIPTION = "transcription"


@dataclass
class Statement:
    """Single statement from participant"""
    statement_id: str
    participant_id: str
    timestamp: datetime
    statement_type: StatementType
    content: str
    
    # Analysis
    key_claims: List[str]
    emotional_tone: str
    stress_level: Optional[float]
    
    # Contradictions
    contradicts_statement_id: Optional[str] = None
    contradiction_confidence: float = 0.0
    contradiction_details: Dict[str, Any] = None


@dataclass
class ContradictionEvent:
    """Contradiction between two statements"""
    contradiction_id: str
    statement_a_id: str
    statement_b_id: str
    participant_id: str
    
    # Timeline
    statement_a_time: datetime
    statement_b_time: datetime
    time_gap_days: float
    
    # Details
    claim_a: str
    claim_b: str
    contradiction_type: str
    confidence: float
    
    # Analysis
    what_changed: str
    possible_reasons: List[str]


class ContradictionTimelineAnalyzer:
    """
    Analyzes and visualizes contradictions in timeline
    """

    def __init__(self):
        """Initialize analyzer"""
        self.statements: Dict[str, Statement] = {}
        self.contradictions: List[ContradictionEvent] = []
        self.contradiction_counter = 0

    def add_statement(
        self,
        statement_id: str,
        participant_id: str,
        timestamp: datetime,
        statement_type: StatementType,
        content: str,
        key_claims: List[str],
        emotional_tone: str = "neutral",
        stress_level: Optional[float] = None
    ) -> Statement:
        """
        Add statement to timeline

        Args:
            statement_id: Unique statement ID
            participant_id: Participant ID
            timestamp: When statement was made
            statement_type: Type of statement
            content: Statement content
            key_claims: Main claims in statement
            emotional_tone: Emotional tone
            stress_level: Stress level (0-100)

        Returns:
            Statement object
        """
        statement = Statement(
            statement_id=statement_id,
            participant_id=participant_id,
            timestamp=timestamp,
            statement_type=statement_type,
            content=content,
            key_claims=key_claims,
            emotional_tone=emotional_tone,
            stress_level=stress_level
        )

        self.statements[statement_id] = statement
        return statement

    def detect_contradictions(
        self,
        participant_id: str,
        contradiction_pairs: List[Tuple[str, str, str, float, str]]
    ):
        """
        Detect contradictions between statements

        Args:
            participant_id: Participant ID
            contradiction_pairs: List of (statement_a_id, statement_b_id, claim_a, claim_b, confidence, contradiction_type)
        """
        for statement_a_id, statement_b_id, claim_a, claim_b, confidence, contradiction_type in contradiction_pairs:
            if statement_a_id not in self.statements or statement_b_id not in self.statements:
                continue

            stmt_a = self.statements[statement_a_id]
            stmt_b = self.statements[statement_b_id]

            # Ensure chronological order
            if stmt_a.timestamp > stmt_b.timestamp:
                stmt_a, stmt_b = stmt_b, stmt_a
                claim_a, claim_b = claim_b, claim_a

            self.contradiction_counter += 1
            contradiction_id = f"CONT_{self.contradiction_counter:06d}"

            time_gap = (stmt_b.timestamp - stmt_a.timestamp).total_seconds() / (24 * 3600)

            contradiction = ContradictionEvent(
                contradiction_id=contradiction_id,
                statement_a_id=stmt_a.statement_id,
                statement_b_id=stmt_b.statement_id,
                participant_id=participant_id,
                statement_a_time=stmt_a.timestamp,
                statement_b_time=stmt_b.timestamp,
                time_gap_days=time_gap,
                claim_a=claim_a,
                claim_b=claim_b,
                contradiction_type=contradiction_type,
                confidence=confidence,
                what_changed=f"Changed from '{claim_a}' to '{claim_b}'",
                possible_reasons=self._generate_possible_reasons(
                    contradiction_type, time_gap, stmt_a.stress_level, stmt_b.stress_level
                )
            )

            self.contradictions.append(contradiction)
            logger.info(f"Detected contradiction: {contradiction_id}")

    def _generate_possible_reasons(
        self,
        contradiction_type: str,
        time_gap_days: float,
        stress_a: Optional[float],
        stress_b: Optional[float]
    ) -> List[str]:
        """Generate possible reasons for contradiction"""
        reasons = []

        if time_gap_days > 30:
            reasons.append("Memory fade over time")
        elif time_gap_days < 1:
            reasons.append("Immediate contradiction - likely deliberate")

        if stress_a and stress_b:
            if stress_b > stress_a * 1.5:
                reasons.append("Increased stress in second statement - may indicate pressure")
            elif stress_a > stress_b * 1.5:
                reasons.append("Decreased stress - may indicate relief or changed story")

        if contradiction_type == "timeline":
            reasons.append("Timeline confusion or deliberate misrepresentation")
        elif contradiction_type == "fact":
            reasons.append("Factual error or deliberate lie")
        elif contradiction_type == "emotion":
            reasons.append("Emotional state change or inconsistent presentation")

        return reasons

    def get_participant_timeline(self, participant_id: str) -> List[Dict[str, Any]]:
        """
        Get chronological timeline for participant

        Args:
            participant_id: Participant ID

        Returns:
            Sorted list of statements with contradiction info
        """
        participant_statements = [
            s for s in self.statements.values()
            if s.participant_id == participant_id
        ]

        # Sort by timestamp
        participant_statements.sort(key=lambda x: x.timestamp)

        timeline = []
        for stmt in participant_statements:
            # Find contradictions involving this statement
            stmt_contradictions = [
                c for c in self.contradictions
                if c.statement_a_id == stmt.statement_id or c.statement_b_id == stmt.statement_id
            ]

            timeline.append({
                'statement_id': stmt.statement_id,
                'timestamp': stmt.timestamp.isoformat(),
                'type': stmt.statement_type.value,
                'content': stmt.content[:200],
                'key_claims': stmt.key_claims,
                'emotional_tone': stmt.emotional_tone,
                'stress_level': stmt.stress_level,
                'contradictions': [
                    {
                        'contradiction_id': c.contradiction_id,
                        'with_statement': c.statement_b_id if c.statement_a_id == stmt.statement_id else c.statement_a_id,
                        'confidence': c.confidence,
                        'type': c.contradiction_type
                    }
                    for c in stmt_contradictions
                ]
            })

        return timeline

    def get_contradiction_summary(self) -> Dict[str, Any]:
        """Get summary of all contradictions"""
        if not self.contradictions:
            return {
                'total_contradictions': 0,
                'by_type': {},
                'critical_contradictions': 0
            }

        by_type = {}
        critical = 0

        for cont in self.contradictions:
            cont_type = cont.contradiction_type
            by_type[cont_type] = by_type.get(cont_type, 0) + 1

            if cont.confidence > 0.9:
                critical += 1

        return {
            'total_contradictions': len(self.contradictions),
            'by_type': by_type,
            'critical_contradictions': critical,
            'average_confidence': sum(c.confidence for c in self.contradictions) / len(self.contradictions) if self.contradictions else 0.0
        }

    def generate_html_timeline(self, participant_id: str, output_path: str) -> bool:
        """
        Generate interactive HTML timeline

        Args:
            participant_id: Participant ID
            output_path: Output file path

        Returns:
            Success status
        """
        try:
            timeline = self.get_participant_timeline(participant_id)

            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Contradiction Timeline - {participant_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .timeline {{ position: relative; padding: 20px 0; }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 4px;
            height: 100%;
            background: #3498db;
        }}
        .timeline-item {{
            margin-bottom: 50px;
            position: relative;
        }}
        .timeline-item:nth-child(odd) .content {{
            margin-left: 0;
            margin-right: 52%;
            text-align: right;
        }}
        .timeline-item:nth-child(even) .content {{
            margin-left: 52%;
            margin-right: 0;
        }}
        .timeline-item::before {{
            content: '';
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 20px;
            background: white;
            border: 4px solid #3498db;
            border-radius: 50%;
            top: 0;
            z-index: 1;
        }}
        .content {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .statement-type {{
            display: inline-block;
            padding: 4px 8px;
            background: #ecf0f1;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        .statement-text {{
            color: #555;
            margin: 10px 0;
            font-style: italic;
        }}
        .contradiction {{
            background: #fff5f5;
            border-left: 4px solid #e74c3c;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }}
        .contradiction-title {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .stress-level {{
            margin-top: 10px;
            padding: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }}
        .stress-high {{ background: #ffe6e6; color: #c0392b; }}
        .stress-medium {{ background: #fff3cd; color: #856404; }}
        .stress-low {{ background: #e8f5e9; color: #2e7d32; }}
    </style>
</head>
<body>
    <h1>Contradiction Timeline: {participant_id}</h1>
    <div class="timeline">
"""

            for i, item in enumerate(timeline):
                timestamp = item['timestamp']
                content = item['content']
                stmt_type = item['type']
                stress = item['stress_level']
                contradictions = item['contradictions']

                stress_class = 'stress-low'
                if stress and stress > 60:
                    stress_class = 'stress-high'
                elif stress and stress > 40:
                    stress_class = 'stress-medium'

                html += f"""        <div class="timeline-item">
            <div class="content">
                <div class="timestamp">{timestamp}</div>
                <span class="statement-type">{stmt_type}</span>
                <div class="statement-text">"{content}..."</div>
"""

                if stress is not None:
                    html += f"""                <div class="stress-level {stress_class}">Stress Level: {stress:.0f}/100</div>
"""

                if contradictions:
                    for cont in contradictions:
                        html += f"""                <div class="contradiction">
                    <div class="contradiction-title">⚠️ Contradicts previous statement</div>
                    <div>Type: {cont['type']}</div>
                    <div>Confidence: {cont['confidence']*100:.0f}%</div>
                </div>
"""

                html += """            </div>
        </div>
"""

            html += """    </div>
</body>
</html>"""

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

            logger.info(f"Generated timeline HTML: {output_path}")
            return True

        except Exception as e:
            logger.error(f"HTML generation error: {e}")
            return False

    def export_timeline_data(self, participant_id: str, output_path: str) -> bool:
        """
        Export timeline data to JSON

        Args:
            participant_id: Participant ID
            output_path: Output file path

        Returns:
            Success status
        """
        try:
            timeline = self.get_participant_timeline(participant_id)
            summary = self.get_contradiction_summary()

            data = {
                'participant_id': participant_id,
                'exported_at': datetime.now().isoformat(),
                'summary': summary,
                'timeline': timeline,
                'contradictions': [
                    {
                        'contradiction_id': c.contradiction_id,
                        'statement_a_id': c.statement_a_id,
                        'statement_b_id': c.statement_b_id,
                        'statement_a_time': c.statement_a_time.isoformat(),
                        'statement_b_time': c.statement_b_time.isoformat(),
                        'time_gap_days': c.time_gap_days,
                        'claim_a': c.claim_a,
                        'claim_b': c.claim_b,
                        'contradiction_type': c.contradiction_type,
                        'confidence': c.confidence,
                        'what_changed': c.what_changed,
                        'possible_reasons': c.possible_reasons
                    }
                    for c in self.contradictions
                    if c.participant_id == participant_id
                ]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported timeline data: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    analyzer = ContradictionTimelineAnalyzer()
    print("Contradiction Timeline Analyzer initialized")
