"""
Evidence Strength Scoring System for Die Waarheid
Rates each piece of evidence for reliability and importance
Enables prioritization of investigative focus

SCORING FACTORS:
- Voice authenticity (Afrikaans verification confidence)
- Timeline consistency (how well it fits the narrative)
- Psychological indicators (stress, defensiveness)
- Cross-reference strength (how many other pieces support it)
- Source reliability (direct vs. indirect evidence)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EvidenceReliability(Enum):
    """Evidence reliability rating"""
    HIGHLY_RELIABLE = "highly_reliable"
    RELIABLE = "reliable"
    MODERATE = "moderate"
    QUESTIONABLE = "questionable"
    UNRELIABLE = "unreliable"


class EvidenceImportance(Enum):
    """Evidence importance rating"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EvidenceScore:
    """Score for single evidence item"""
    evidence_id: str
    
    # Component scores (0-100)
    authenticity_score: float
    timeline_consistency_score: float
    psychological_indicator_score: float
    cross_reference_score: float
    source_reliability_score: float
    
    # Overall scores
    reliability_score: float
    importance_score: float
    overall_strength: float
    
    # Ratings
    reliability_rating: EvidenceReliability
    importance_rating: EvidenceImportance
    
    # Details
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'evidence_id': self.evidence_id,
            'authenticity_score': self.authenticity_score,
            'timeline_consistency_score': self.timeline_consistency_score,
            'psychological_indicator_score': self.psychological_indicator_score,
            'cross_reference_score': self.cross_reference_score,
            'source_reliability_score': self.source_reliability_score,
            'reliability_score': self.reliability_score,
            'importance_score': self.importance_score,
            'overall_strength': self.overall_strength,
            'reliability_rating': self.reliability_rating.value,
            'importance_rating': self.importance_rating.value,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'recommendations': self.recommendations
        }


class EvidenceScoringSystem:
    """
    Evidence strength scoring system
    Rates and prioritizes evidence
    """

    def __init__(self):
        """Initialize scoring system"""
        self.scores: Dict[str, EvidenceScore] = {}

    def score_evidence(
        self,
        evidence_id: str,
        evidence_type: str,
        afrikaans_confidence: Optional[float] = None,
        timeline_consistency: Optional[float] = None,
        stress_level: Optional[float] = None,
        baseline_stress: Optional[float] = None,
        cross_references: Optional[int] = None,
        contradictions: Optional[int] = None,
        forensic_flags: Optional[List[str]] = None,
        expert_findings: Optional[int] = None
    ) -> EvidenceScore:
        """
        Score evidence based on multiple factors

        Args:
            evidence_id: Evidence identifier
            evidence_type: Type of evidence (chat_export, voice_note, etc.)
            afrikaans_confidence: Afrikaans verification confidence (0-1)
            timeline_consistency: How well it fits timeline (0-1)
            stress_level: Current stress level (0-100)
            baseline_stress: Baseline stress level (0-100)
            cross_references: Number of cross-references
            contradictions: Number of contradictions found
            forensic_flags: List of forensic flags
            expert_findings: Number of expert findings

        Returns:
            Evidence score
        """
        logger.info(f"Scoring evidence: {evidence_id}")

        # Initialize component scores
        authenticity_score = 50.0
        timeline_consistency_score = 50.0
        psychological_indicator_score = 50.0
        cross_reference_score = 50.0
        source_reliability_score = 50.0

        strengths = []
        weaknesses = []

        # 1. AUTHENTICITY SCORE (Voice/Afrikaans verification)
        if evidence_type == "voice_note" or evidence_type == "transcription":
            if afrikaans_confidence is not None:
                authenticity_score = afrikaans_confidence * 100
                
                if afrikaans_confidence > 0.9:
                    strengths.append("Excellent Afrikaans authenticity verification")
                elif afrikaans_confidence > 0.7:
                    strengths.append("Good Afrikaans verification")
                elif afrikaans_confidence > 0.5:
                    weaknesses.append("Moderate Afrikaans verification - human review recommended")
                else:
                    weaknesses.append("Low Afrikaans authenticity - requires human verification")
            else:
                authenticity_score = 75.0  # Default for voice notes
                strengths.append("Voice note - direct audio evidence")
        else:
            authenticity_score = 60.0  # Text evidence lower authenticity
            weaknesses.append("Text-based evidence - no voice verification available")

        # 2. TIMELINE CONSISTENCY SCORE
        if timeline_consistency is not None:
            timeline_consistency_score = timeline_consistency * 100
            
            if timeline_consistency > 0.9:
                strengths.append("Excellent timeline consistency")
            elif timeline_consistency > 0.7:
                strengths.append("Good timeline fit")
            elif timeline_consistency > 0.5:
                weaknesses.append("Moderate timeline inconsistencies")
            else:
                weaknesses.append("Significant timeline inconsistencies - potential contradiction")
        else:
            timeline_consistency_score = 50.0

        # 3. PSYCHOLOGICAL INDICATOR SCORE
        if stress_level is not None and baseline_stress is not None and baseline_stress > 0:
            stress_ratio = stress_level / baseline_stress
            
            if stress_ratio < 1.2:
                psychological_indicator_score = 80.0
                strengths.append("Consistent stress levels - credible")
            elif stress_ratio < 1.5:
                psychological_indicator_score = 70.0
                strengths.append("Slightly elevated stress - normal variation")
            elif stress_ratio < 2.0:
                psychological_indicator_score = 50.0
                weaknesses.append("Elevated stress levels - may indicate deception or pressure")
            else:
                psychological_indicator_score = 30.0
                weaknesses.append("Significantly elevated stress - strong indicator of concern")
        else:
            psychological_indicator_score = 60.0

        # 4. CROSS-REFERENCE SCORE
        if cross_references is not None:
            # More cross-references = stronger evidence
            if cross_references >= 5:
                cross_reference_score = 95.0
                strengths.append(f"Strongly corroborated by {cross_references} other evidence items")
            elif cross_references >= 3:
                cross_reference_score = 80.0
                strengths.append(f"Corroborated by {cross_references} other evidence items")
            elif cross_references >= 1:
                cross_reference_score = 65.0
                strengths.append(f"Supported by {cross_references} other evidence item(s)")
            else:
                cross_reference_score = 40.0
                weaknesses.append("No corroborating evidence found")
        else:
            cross_reference_score = 50.0

        # 5. SOURCE RELIABILITY SCORE
        if forensic_flags:
            flag_count = len(forensic_flags)
            if flag_count == 0:
                source_reliability_score = 90.0
                strengths.append("No forensic flags - clean evidence")
            elif flag_count == 1:
                source_reliability_score = 70.0
                weaknesses.append(f"One forensic flag: {forensic_flags[0]}")
            elif flag_count <= 3:
                source_reliability_score = 50.0
                weaknesses.append(f"Multiple forensic flags ({flag_count}) - reliability questionable")
            else:
                source_reliability_score = 30.0
                weaknesses.append(f"Many forensic flags ({flag_count}) - significant reliability concerns")
        else:
            source_reliability_score = 70.0

        # Calculate overall scores
        reliability_score = (
            authenticity_score * 0.3 +
            timeline_consistency_score * 0.25 +
            psychological_indicator_score * 0.2 +
            cross_reference_score * 0.15 +
            source_reliability_score * 0.1
        )

        # Importance score based on expert findings and contradictions
        importance_score = 50.0
        
        if expert_findings is not None:
            if expert_findings >= 5:
                importance_score = 95.0
                strengths.append(f"Critical: {expert_findings} expert findings")
            elif expert_findings >= 3:
                importance_score = 80.0
                strengths.append(f"Important: {expert_findings} expert findings")
            elif expert_findings >= 1:
                importance_score = 65.0
            else:
                importance_score = 40.0

        if contradictions is not None and contradictions > 0:
            importance_score = min(100, importance_score + (contradictions * 15))
            strengths.append(f"Contains {contradictions} contradiction(s) - high importance")

        # Determine ratings
        if reliability_score >= 80:
            reliability_rating = EvidenceReliability.HIGHLY_RELIABLE
        elif reliability_score >= 65:
            reliability_rating = EvidenceReliability.RELIABLE
        elif reliability_score >= 50:
            reliability_rating = EvidenceReliability.MODERATE
        elif reliability_score >= 35:
            reliability_rating = EvidenceReliability.QUESTIONABLE
        else:
            reliability_rating = EvidenceReliability.UNRELIABLE

        if importance_score >= 80:
            importance_rating = EvidenceImportance.CRITICAL
        elif importance_score >= 65:
            importance_rating = EvidenceImportance.HIGH
        elif importance_score >= 50:
            importance_rating = EvidenceImportance.MEDIUM
        else:
            importance_rating = EvidenceImportance.LOW

        # Overall strength (combination of reliability and importance)
        overall_strength = (reliability_score * 0.6) + (importance_score * 0.4)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            reliability_rating, importance_rating, strengths, weaknesses
        )

        score = EvidenceScore(
            evidence_id=evidence_id,
            authenticity_score=authenticity_score,
            timeline_consistency_score=timeline_consistency_score,
            psychological_indicator_score=psychological_indicator_score,
            cross_reference_score=cross_reference_score,
            source_reliability_score=source_reliability_score,
            reliability_score=reliability_score,
            importance_score=importance_score,
            overall_strength=overall_strength,
            reliability_rating=reliability_rating,
            importance_rating=importance_rating,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )

        self.scores[evidence_id] = score
        return score

    def _generate_recommendations(
        self,
        reliability: EvidenceReliability,
        importance: EvidenceImportance,
        strengths: List[str],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []

        # Reliability-based recommendations
        if reliability == EvidenceReliability.HIGHLY_RELIABLE:
            recommendations.append("Use as primary evidence in investigation")
        elif reliability == EvidenceReliability.RELIABLE:
            recommendations.append("Use as supporting evidence")
        elif reliability == EvidenceReliability.MODERATE:
            recommendations.append("Corroborate with additional evidence before using")
        elif reliability == EvidenceReliability.QUESTIONABLE:
            recommendations.append("Verify authenticity before relying on this evidence")
        else:
            recommendations.append("Do not use as primary evidence - requires significant corroboration")

        # Importance-based recommendations
        if importance == EvidenceImportance.CRITICAL:
            recommendations.append("Prioritize investigation focus on this evidence")
            recommendations.append("Ensure thorough expert analysis")
        elif importance == EvidenceImportance.HIGH:
            recommendations.append("Include in core investigation narrative")
        elif importance == EvidenceImportance.MEDIUM:
            recommendations.append("Include in supporting evidence")

        # Weakness-based recommendations
        if weaknesses:
            if any("timeline" in w.lower() for w in weaknesses):
                recommendations.append("Verify timeline with independent sources")
            if any("stress" in w.lower() for w in weaknesses):
                recommendations.append("Investigate cause of elevated stress")
            if any("flag" in w.lower() for w in weaknesses):
                recommendations.append("Address forensic flags before using as primary evidence")

        return recommendations

    def get_top_evidence(self, limit: int = 10) -> List[EvidenceScore]:
        """Get top evidence by overall strength"""
        sorted_scores = sorted(
            self.scores.values(),
            key=lambda x: x.overall_strength,
            reverse=True
        )
        return sorted_scores[:limit]

    def get_critical_evidence(self) -> List[EvidenceScore]:
        """Get all critical importance evidence"""
        return [
            s for s in self.scores.values()
            if s.importance_rating == EvidenceImportance.CRITICAL
        ]

    def get_unreliable_evidence(self) -> List[EvidenceScore]:
        """Get unreliable evidence"""
        return [
            s for s in self.scores.values()
            if s.reliability_rating == EvidenceReliability.UNRELIABLE
        ]

    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get summary of all evidence scores"""
        if not self.scores:
            return {'total_evidence': 0}

        scores_list = list(self.scores.values())
        
        critical = len([s for s in scores_list if s.importance_rating == EvidenceImportance.CRITICAL])
        high = len([s for s in scores_list if s.importance_rating == EvidenceImportance.HIGH])
        reliable = len([s for s in scores_list if s.reliability_rating in [
            EvidenceReliability.HIGHLY_RELIABLE, EvidenceReliability.RELIABLE
        ]])
        unreliable = len([s for s in scores_list if s.reliability_rating == EvidenceReliability.UNRELIABLE])

        avg_strength = sum(s.overall_strength for s in scores_list) / len(scores_list) if scores_list else 0.0
        avg_reliability = sum(s.reliability_score for s in scores_list) / len(scores_list) if scores_list else 0.0

        return {
            'total_evidence': len(scores_list),
            'critical_importance': critical,
            'high_importance': high,
            'reliable_evidence': reliable,
            'unreliable_evidence': unreliable,
            'average_strength': avg_strength,
            'average_reliability': avg_reliability,
            'top_evidence': [s.evidence_id for s in self.get_top_evidence(5)]
        }


if __name__ == "__main__":
    system = EvidenceScoringSystem()
    print("Evidence Scoring System initialized")
