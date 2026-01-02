"""
Toxicity and Gaslighting Detection Module
Analyzes text for toxic patterns, gaslighting phrases, and narcissistic behaviors
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class ToxicityResult:
    """Result of toxicity analysis"""
    toxicity_score: float  # 0-100
    gaslighting_score: float  # 0-100
    narcissistic_score: float  # 0-100
    detected_phrases: List[str]
    category: str
    severity: str  # low, medium, high, severe

class ToxicityDetector:
    """Detects toxic communication patterns"""
    
    def __init__(self):
        # Gaslighting phrases from config
        self.gaslighting_phrases = [
            "never said that", "you're crazy", "imagining things",
            "overreacting", "too sensitive", "making things up",
            "didn't happen", "in your head", "being dramatic",
            "jy dink te veel", "jy is mal", "dit het nooit gebeur nie",
            "you're remembering it wrong", "that's not what happened",
            "you're confused", "you're exaggerating", "you're paranoid"
        ]
        
        # Toxicity phrases
        self.toxicity_phrases = [
            "stupid", "hate", "idiot", "shut up", "dumb", "useless",
            "pathetic", "loser", "worthless", "disgusting",
            "dom", "haat", "idioot", "hou jou bek", "nutteloos",
            "kill yourself", "die", "worthless piece of", "garbage"
        ]
        
        # Narcissistic patterns
        self.narcissistic_patterns = [
            "I'm the victim", "you made me", "look what you made me do",
            "everyone agrees with me", "nobody likes you",
            "ek is die slagoffer", "jy het my gemaak",
            "I'm always right", "you're wrong", "it's all your fault",
            "you're lucky to have me", "no one else would want you"
        ]
        
        # Evasion patterns
        self.evasion_patterns = [
            "maybe", "perhaps", "i think", "not sure", "probably",
            "i don't recall", "i don't remember", "i'm not certain"
        ]
        
        # Contradiction indicators
        self.contradiction_words = ["but", "however", "although", "though"]
        
    def analyze_message(self, message: str) -> ToxicityResult:
        """
        Analyze a single message for toxicity patterns
        
        Args:
            message: The message text to analyze
            
        Returns:
            ToxicityResult with scores and detected patterns
        """
        if not message:
            return ToxicityResult(0, 0, 0, [], "neutral", "low")
        
        message_lower = message.lower()
        detected_phrases = []
        
        # Check gaslighting
        gaslighting_score = 0
        for phrase in self.gaslighting_phrases:
            if phrase in message_lower:
                gaslighting_score += 20
                detected_phrases.append(f"Gaslighting: '{phrase}'")
        
        # Check toxicity
        toxicity_score = 0
        for phrase in self.toxicity_phrases:
            if phrase in message_lower:
                toxicity_score += 25
                detected_phrases.append(f"Toxic: '{phrase}'")
        
        # Check narcissistic patterns
        narcissistic_score = 0
        for pattern in self.narcissistic_patterns:
            if pattern in message_lower:
                narcissistic_score += 15
                detected_phrases.append(f"Narcissistic: '{pattern}'")
        
        # Check evasion
        evasion_count = sum(1 for word in self.evasion_patterns if word in message_lower)
        if evasion_count > 2:
            toxicity_score += 10
            detected_phrases("High evasion detected")
        
        # Calculate overall score
        overall_score = max(toxicity_score, gaslighting_score, narcissistic_score)
        
        # Determine category and severity
        if gaslighting_score > 0:
            category = "gaslighting"
        elif toxicity_score > 0:
            category = "toxic"
        elif narcissistic_score > 0:
            category = "narcissistic"
        else:
            category = "neutral"
        
        if overall_score >= 75:
            severity = "severe"
        elif overall_score >= 50:
            severity = "high"
        elif overall_score >= 25:
            severity = "medium"
        else:
            severity = "low"
        
        return ToxicityResult(
            toxicity_score=min(toxicity_score, 100),
            gaslighting_score=min(gaslighting_score, 100),
            narcissistic_score=min(narcissistic_score, 100),
            detected_phrases=detected_phrases,
            category=category,
            severity=severity
        )
    
    def analyze_conversation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze entire conversation for patterns
        
        Args:
            df: DataFrame with messages
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'total_messages': len(df),
            'toxic_messages': 0,
            'gaslighting_messages': 0,
            'narcissistic_messages': 0,
            'severity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'severe': 0},
            'top_offenders': [],
            'pattern_timeline': []
        }
        
        # Track sender statistics
        sender_stats = {}
        
        for idx, row in df.iterrows():
            message = str(row.get('message', ''))
            sender = row.get('sender', 'Unknown')
            
            analysis = self.analyze_message(message)
            
            # Update counts
            if analysis.toxicity_score > 0:
                results['toxic_messages'] += 1
            if analysis.gaslighting_score > 0:
                results['gaslighting_messages'] += 1
            if analysis.narcissistic_score > 0:
                results['narcissistic_messages'] += 1
            
            # Update severity distribution
            results['severity_distribution'][analysis.severity] += 1
            
            # Track sender statistics
            if sender not in sender_stats:
                sender_stats[sender] = {
                    'total': 0,
                    'toxic': 0,
                    'gaslighting': 0,
                    'narcissistic': 0,
                    'max_severity': 'low'
                }
            
            sender_stats[sender]['total'] += 1
            if analysis.toxicity_score > 0:
                sender_stats[sender]['toxic'] += 1
            if analysis.gaslighting_score > 0:
                sender_stats[sender]['gaslighting'] += 1
            if analysis.narcissistic_score > 0:
                sender_stats[sender]['narcissistic'] += 1
            
            # Update max severity
            severity_order = ['low', 'medium', 'high', 'severe']
            current_idx = severity_order.index(sender_stats[sender]['max_severity'])
            new_idx = severity_order.index(analysis.severity)
            if new_idx > current_idx:
                sender_stats[sender]['max_severity'] = analysis.severity
            
            # Add to timeline if significant
            if analysis.severity in ['high', 'severe']:
                results['pattern_timeline'].append({
                    'timestamp': row.get('timestamp'),
                    'sender': sender,
                    'severity': analysis.severity,
                    'category': analysis.category,
                    'phrase': analysis.detected_phrases[0] if analysis.detected_phrases else ''
                })
        
        # Calculate top offenders
        for sender, stats in sender_stats.items():
            toxicity_rate = (stats['toxic'] + stats['gaslighting'] + stats['narcissistic']) / stats['total'] * 100
            results['top_offenders'].append({
                'sender': sender,
                'toxicity_rate': toxicity_rate,
                'max_severity': stats['max_severity'],
                'toxic_count': stats['toxic'],
                'gaslighting_count': stats['gaslighting'],
                'narcissistic_count': stats['narcissistic']
            })
        
        # Sort by toxicity rate
        results['top_offenders'].sort(key=lambda x: x['toxicity_rate'], reverse=True)
        
        return results
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate a human-readable report"""
        report = []
        
        report.append("## 🚨 Toxicity Analysis Report\n")
        
        # Summary
        total = analysis['total_messages']
        toxic_pct = (analysis['toxic_messages'] / total * 100) if total > 0 else 0
        gaslighting_pct = (analysis['gaslighting_messages'] / total * 100) if total > 0 else 0
        
        report.append(f"**Total Messages Analyzed:** {total}")
        report.append(f"**Toxic Messages:** {analysis['toxic_messages']} ({toxic_pct:.1f}%)")
        report.append(f"**Gaslighting Instances:** {analysis['gaslighting_messages']} ({gaslighting_pct:.1f}%)")
        report.append(f"**Narcissistic Patterns:** {analysis['narcissistic_messages']}")
        
        # Severity breakdown
        report.append("\n### 📊 Severity Distribution")
        for severity, count in analysis['severity_distribution'].items():
            if count > 0:
                report.append(f"- {severity.capitalize()}: {count} messages")
        
        # Top offenders
        if analysis['top_offenders']:
            report.append("\n### 👥 Top Offenders")
            for offender in analysis['top_offenders'][:5]:
                report.append(f"**{offender['sender']}:** {offender['toxicity_rate']:.1f}% toxicity rate")
                report.append(f"  - Toxic: {offender['toxic_count']}")
                report.append(f"  - Gaslighting: {offender['gaslighting_count']}")
                report.append(f"  - Narcissistic: {offender['narcissistic_count']}")
        
        # Timeline of severe incidents
        if analysis['pattern_timeline']:
            report.append("\n### ⚠️ Critical Incidents Timeline")
            for incident in analysis['pattern_timeline'][:10]:
                report.append(f"**{incident['timestamp']}** - {incident['sender']}: {incident['category']} ({incident['severity']})")
        
        return "\n".join(report)

# Global instance
toxicity_detector = ToxicityDetector()
