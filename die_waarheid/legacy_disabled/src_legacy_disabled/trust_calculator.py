"""
Trust Score Calculator for Die Waarheid
Calculates overall trustworthiness based on multiple forensic factors
"""

from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta

class TrustScoreCalculator:
    """Calculates trust scores based on forensic analysis"""
    
    def __init__(self):
        self.weights = {
            'consistency': 0.25,      # Message consistency
            'stress_patterns': 0.20,  # Voice stress analysis
            'transparency': 0.20,     # Evasion and contradiction
            'toxicity': 0.15,         # Toxic behavior
            'temporal_patterns': 0.10, # Timing anomalies
            'verification': 0.10      # Claims vs evidence
        }
    
    def calculate_overall_trust(self, df: pd.DataFrame, toxicity_analysis: Dict, 
                               audio_analysis: Dict) -> Dict:
        """
        Calculate overall trust score for the conversation
        
        Args:
            df: DataFrame with all messages
            toxicity_analysis: Results from toxicity detector
            audio_analysis: Audio forensic results
            
        Returns:
            Dictionary with trust scores and breakdown
        """
        scores = {}
        
        # 1. Consistency Score (0-100)
        scores['consistency'] = self._calculate_consistency_score(df)
        
        # 2. Stress Patterns Score (0-100)
        scores['stress_patterns'] = self._calculate_stress_score(audio_analysis)
        
        # 3. Transparency Score (0-100)
        scores['transparency'] = self._calculate_transparency_score(df)
        
        # 4. Toxicity Score (0-100, inverted - high toxicity = low trust)
        raw_toxicity = self._calculate_toxicity_score(toxicity_analysis)
        scores['toxicity'] = max(0, 100 - raw_toxicity)
        
        # 5. Temporal Patterns Score (0-100)
        scores['temporal_patterns'] = self._calculate_temporal_score(df)
        
        # 6. Verification Score (0-100)
        scores['verification'] = self._calculate_verification_score(df)
        
        # Calculate weighted overall score
        overall_score = sum(scores[factor] * self.weights[factor] for factor in scores)
        
        # Determine trust level
        if overall_score >= 80:
            trust_level = "Very High"
            interpretation = "Communication shows high consistency and transparency"
        elif overall_score >= 60:
            trust_level = "High"
            interpretation = "Generally trustworthy with minor concerns"
        elif overall_score >= 40:
            trust_level = "Medium"
            interpretation = "Some inconsistencies detected, verify important claims"
        elif overall_score >= 20:
            trust_level = "Low"
            interpretation = "Significant trust issues, exercise caution"
        else:
            trust_level = "Very Low"
            interpretation = "High deception indicators, not trustworthy"
        
        return {
            'overall_score': round(overall_score, 1),
            'trust_level': trust_level,
            'interpretation': interpretation,
            'breakdown': scores,
            'recommendations': self._generate_recommendations(scores, overall_score)
        }
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency based on contradictions and message patterns"""
        if df.empty:
            return 50
        
        # Count contradictions
        contradictions = 0
        total_messages = len(df)
        
        for _, row in df.iterrows():
            message = str(row.get('message', '')).lower()
            
            # Look for contradiction indicators
            if any(word in message for word in ['but actually', 'wait i', 'correction', 'i meant']):
                contradictions += 1
        
        # Calculate score (fewer contradictions = higher score)
        contradiction_rate = contradictions / total_messages if total_messages > 0 else 0
        base_score = max(0, 100 - (contradiction_rate * 100))
        
        # Adjust for message length consistency
        avg_length = df['message'].str.len().mean()
        length_variance = df['message'].str.len().std()
        
        # High variance in message length can indicate evasiveness
        if length_variance > avg_length:
            base_score -= 10
        
        return max(0, min(100, base_score))
    
    def _calculate_stress_score(self, audio_analysis: Dict) -> float:
        """Calculate score based on voice stress patterns"""
        if not audio_analysis or 'stress_distribution' not in audio_analysis:
            return 50
        
        stress_dist = audio_analysis['stress_distribution']
        total_audio = sum(stress_dist.values())
        
        if total_audio == 0:
            return 50
        
        # Calculate weighted stress score
        high_stress = stress_dist.get('High', 0) + stress_dist.get('Very High', 0)
        medium_stress = stress_dist.get('Medium', 0)
        low_stress = stress_dist.get('Low', 0) + stress_dist.get('Very Low', 0)
        
        # High stress reduces trust score
        stress_penalty = (high_stress / total_audio * 50) + (medium_stress / total_audio * 20)
        base_score = max(0, 100 - stress_penalty)
        
        # Bonus for consistently low stress
        if low_stress / total_audio > 0.7:
            base_score = min(100, base_score + 10)
        
        return base_score
    
    def _calculate_transparency_score(self, df: pd.DataFrame) -> float:
        """Calculate transparency based on evasion and directness"""
        if df.empty:
            return 50
        
        evasion_words = ['maybe', 'perhaps', 'i think', 'not sure', 'probably', 'might']
        direct_words = ['yes', 'no', 'definitely', 'certainly', 'absolutely']
        
        evasion_count = 0
        direct_count = 0
        total_words = 0
        
        for _, row in df.iterrows():
            message = str(row.get('message', '')).lower().split()
            total_words += len(message)
            
            evasion_count += sum(1 for word in message if word in evasion_words)
            direct_count += sum(1 for word in message if word in direct_words)
        
        if total_words == 0:
            return 50
        
        evasion_rate = evasion_count / total_words
        directness_rate = direct_count / total_words
        
        # Calculate score (high directness = high score, high evasion = low score)
        base_score = 50 + (directness_rate * 100) - (evasion_rate * 100)
        
        return max(0, min(100, base_score))
    
    def _calculate_toxicity_score(self, toxicity_analysis: Dict) -> float:
        """Calculate toxicity impact on trust"""
        total_messages = toxicity_analysis.get('total_messages', 1)
        toxic_messages = toxicity_analysis.get('toxic_messages', 0)
        gaslighting_messages = toxicity_analysis.get('gaslighting_messages', 0)
        
        # Gaslighting is more severe than general toxicity
        toxicity_rate = (toxic_messages / total_messages * 100)
        gaslighting_penalty = (gaslighting_messages / total_messages * 150)
        
        return min(100, toxicity_rate + gaslighting_penalty)
    
    def _calculate_temporal_score(self, df: pd.DataFrame) -> float:
        """Analyze timing patterns for suspicious behavior"""
        if 'timestamp' not in df.columns or df.empty:
            return 50
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate response times
        response_times = []
        for sender in df['sender'].unique():
            sender_messages = df[df['sender'] == sender]
            if len(sender_messages) > 1:
                time_diffs = sender_messages['timestamp'].diff().dt.total_seconds() / 60  # minutes
                # Remove very long gaps (sleep periods)
                time_diffs = time_diffs[time_diffs < 240]  # Less than 4 hours
                response_times.extend(time_diffs.tolist())
        
        if not response_times:
            return 50
        
        # Analyze response patterns
        avg_response = sum(response_times) / len(response_times)
        
        # Very fast responses can indicate pre-planned messages
        very_fast = sum(1 for t in response_times if t < 0.5)  # Less than 30 seconds
        very_slow = sum(1 for t in response_times if t > 60)  # More than 1 hour
        
        fast_penalty = (very_fast / len(response_times)) * 30
        slow_penalty = (very_slow / len(response_times)) * 20
        
        base_score = 100 - fast_penalty - slow_penalty
        
        return max(0, min(100, base_score))
    
    def _calculate_verification_score(self, df: pd.DataFrame) -> float:
        """Score based on claims vs available evidence"""
        if df.empty:
            return 50
        
        # Look for claims that should have evidence
        claim_indicators = ['i promise', 'i swear', 'honestly', 'truthfully', 'believe me']
        evidence_indicators = ['here is', 'attached', 'screenshot', 'proof', 'see for yourself']
        
        claims = 0
        evidence_provided = 0
        
        for _, row in df.iterrows():
            message = str(row.get('message', '')).lower()
            
            if any(indicator in message for indicator in claim_indicators):
                claims += 1
                # Check if evidence is provided in the same or next message
                if any(indicator in message for indicator in evidence_indicators):
                    evidence_provided += 1
        
        if claims == 0:
            return 50
        
        # Score based on evidence to claim ratio
        evidence_ratio = evidence_provided / claims
        base_score = 50 + (evidence_ratio * 50)
        
        return max(0, min(100, base_score))
    
    def _generate_recommendations(self, scores: Dict, overall_score: float) -> List[str]:
        """Generate recommendations based on score breakdown"""
        recommendations = []
        
        if scores['consistency'] < 50:
            recommendations.append("⚠️ Watch for contradictions in statements")
        
        if scores['stress_patterns'] < 50:
            recommendations.append("🔊 High stress detected in voice notes - may indicate deception")
        
        if scores['transparency'] < 50:
            recommendations.append("💬 Evasive language detected - ask direct questions")
        
        if scores['toxicity'] < 70:
            recommendations.append("🚫 Toxic communication patterns present - maintain boundaries")
        
        if scores['temporal_patterns'] < 50:
            recommendations.append("⏰ Unusual message timing patterns detected")
        
        if scores['verification'] < 50:
            recommendations.append("🔍 Claims made without supporting evidence")
        
        if overall_score < 40:
            recommendations.append("🛑 Overall trust score is low - exercise extreme caution")
        
        if not recommendations:
            recommendations.append("✅ Communication patterns appear trustworthy")
        
        return recommendations
    
    def calculate_sender_trust(self, df: pd.DataFrame, sender: str) -> Dict:
        """Calculate trust score for a specific sender"""
        sender_df = df[df['sender'] == sender]
        
        if sender_df.empty:
            return {'score': 50, 'level': 'Unknown', 'issues': []}
        
        # Analyze sender-specific patterns
        issues = []
        
        # Check message consistency
        contradictions = 0
        for _, row in sender_df.iterrows():
            message = str(row.get('message', '')).lower()
            if any(word in message for word in ['but actually', 'wait i', 'correction']):
                contradictions += 1
        
        if contradictions > len(sender_df) * 0.1:
            issues.append("Frequent contradictions")
        
        # Check evasion
        evasion_count = 0
        for _, row in sender_df.iterrows():
            message = str(row.get('message', '')).lower()
            if any(word in message for word in ['maybe', 'perhaps', 'not sure']):
                evasion_count += 1
        
        if evasion_count > len(sender_df) * 0.3:
            issues.append("High evasion rate")
        
        # Calculate score
        base_score = 100
        base_score -= (contradictions / len(sender_df) * 100)
        base_score -= (evasion_count / len(sender_df) * 50)
        
        score = max(0, min(100, base_score))
        
        if score >= 70:
            level = "Trustworthy"
        elif score >= 50:
            level = "Cautious"
        elif score >= 30:
            level = "Suspicious"
        else:
            level = "Untrustworthy"
        
        return {
            'score': round(score, 1),
            'level': level,
            'issues': issues,
            'message_count': len(sender_df)
        }

# Global instance
trust_calculator = TrustScoreCalculator()
