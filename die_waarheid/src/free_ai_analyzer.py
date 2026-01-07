"""
Free AI Analyzer for Die Waarheid using Hugging Face Transformers
Provides psychological profiling and pattern detection without API keys
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from functools import lru_cache
import warnings

# Suppress transformer warnings
warnings.filterwarnings("ignore", message=".*transformers.*")
warnings.filterwarnings("ignore", message=".*torch.*")

try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForCausalLM
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config import (
    GASLIGHTING_PHRASES,
    TOXICITY_PHRASES,
    NARCISSISTIC_PATTERNS
)

logger = logging.getLogger(__name__)


class FreeAIAnalyzer:
    """
    Free AI analyzer using Hugging Face Transformers
    Provides psychological analysis without requiring API keys
    """
    
    def __init__(self):
        """Initialize the free AI analyzer"""
        self.configured = False
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Install with: pip install transformers torch")
            return
            
        try:
            self._initialize_models()
            self.configured = True
            logger.info(f"Free AI Analyzer initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Free AI Analyzer: {e}")
            self.configured = False
    
    def _initialize_models(self):
        """Initialize all required models"""
        logger.info("Loading AI models (this may take a few minutes on first run)...")
        
        # Emotion analysis model
        try:
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Emotion analysis model loaded")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
        
        # Sentiment analysis model
        try:
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Sentiment analysis model loaded")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
        
        # Toxicity detection model
        try:
            self.models['toxicity'] = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Toxicity detection model loaded")
        except Exception as e:
            logger.warning(f"Failed to load toxicity model: {e}")
        
        # Text generation for psychological profiling
        try:
            self.models['generation'] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            logger.info("✅ Text generation model loaded")
        except Exception as e:
            logger.warning(f"Failed to load generation model: {e}")
    
    def analyze_message(self, text: str) -> Dict:
        """
        Analyze a single message for psychological indicators
        
        Args:
            text: Message text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not self.configured:
            return {
                'success': False,
                'message': 'Free AI Analyzer not configured'
            }
        
        try:
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'analysis': {}
            }
            
            # Emotion analysis
            if 'emotion' in self.models:
                emotion_result = self.models['emotion'](text)
                result['analysis']['emotion'] = {
                    'primary_emotion': emotion_result[0]['label'].lower(),
                    'confidence': emotion_result[0]['score'],
                    'all_emotions': emotion_result
                }
            
            # Sentiment analysis
            if 'sentiment' in self.models:
                sentiment_result = self.models['sentiment'](text)
                result['analysis']['sentiment'] = {
                    'label': sentiment_result[0]['label'].lower(),
                    'confidence': sentiment_result[0]['score'],
                    'polarity': self._convert_sentiment_to_polarity(sentiment_result[0])
                }
            
            # Toxicity analysis
            if 'toxicity' in self.models:
                toxicity_result = self.models['toxicity'](text)
                result['analysis']['toxicity'] = {
                    'is_toxic': toxicity_result[0]['label'] == 'TOXIC',
                    'toxicity_score': toxicity_result[0]['score'] if toxicity_result[0]['label'] == 'TOXIC' else 1 - toxicity_result[0]['score'],
                    'confidence': toxicity_result[0]['score']
                }
            
            # Pattern-based analysis (gaslighting, narcissism)
            result['analysis']['patterns'] = self._analyze_patterns(text)
            
            # Aggression level estimation
            result['analysis']['aggression'] = self._estimate_aggression(text, result['analysis'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            return {
                'success': False,
                'message': f'Analysis failed: {str(e)}'
            }
    
    def _convert_sentiment_to_polarity(self, sentiment_result: Dict) -> str:
        """Convert sentiment result to simple polarity"""
        label = sentiment_result['label'].lower()
        if 'positive' in label:
            return 'positive'
        elif 'negative' in label:
            return 'negative'
        else:
            return 'neutral'
    
    def _analyze_patterns(self, text: str) -> Dict:
        """Analyze text for psychological patterns"""
        text_lower = text.lower()
        
        # Gaslighting detection
        gaslighting_matches = []
        for phrase in GASLIGHTING_PHRASES:
            if phrase.lower() in text_lower:
                gaslighting_matches.append(phrase)
        
        # Toxicity phrases
        toxicity_matches = []
        for phrase in TOXICITY_PHRASES:
            if phrase.lower() in text_lower:
                toxicity_matches.append(phrase)
        
        # Narcissistic patterns
        narcissistic_matches = []
        for pattern in NARCISSISTIC_PATTERNS:
            if pattern.lower() in text_lower:
                narcissistic_matches.append(pattern)
        
        return {
            'gaslighting': {
                'detected': len(gaslighting_matches) > 0,
                'matches': gaslighting_matches,
                'score': min(1.0, len(gaslighting_matches) / 3.0)
            },
            'toxicity_phrases': {
                'detected': len(toxicity_matches) > 0,
                'matches': toxicity_matches,
                'score': min(1.0, len(toxicity_matches) / 5.0)
            },
            'narcissistic': {
                'detected': len(narcissistic_matches) > 0,
                'matches': narcissistic_matches,
                'score': min(1.0, len(narcissistic_matches) / 3.0)
            }
        }
    
    def _estimate_aggression(self, text: str, analysis: Dict) -> Dict:
        """Estimate aggression level based on multiple factors"""
        aggression_score = 0.0
        
        # Factor in toxicity
        if 'toxicity' in analysis:
            aggression_score += analysis['toxicity'].get('toxicity_score', 0) * 0.4
        
        # Factor in negative sentiment
        if 'sentiment' in analysis:
            if analysis['sentiment']['polarity'] == 'negative':
                aggression_score += analysis['sentiment']['confidence'] * 0.3
        
        # Factor in patterns
        if 'patterns' in analysis:
            aggression_score += analysis['patterns']['toxicity_phrases']['score'] * 0.3
        
        # Text-based indicators
        text_lower = text.lower()
        aggressive_indicators = [
            '!', 'shut up', 'stupid', 'idiot', 'hate', 'kill', 'die',
            'fuck', 'damn', 'hell', 'bitch', 'asshole'
        ]
        
        indicator_count = sum(1 for indicator in aggressive_indicators if indicator in text_lower)
        aggression_score += min(0.5, indicator_count / 10.0)
        
        # Normalize score
        aggression_score = min(1.0, aggression_score)
        
        # Determine level
        if aggression_score < 0.3:
            level = 'low'
        elif aggression_score < 0.7:
            level = 'medium'
        else:
            level = 'high'
        
        return {
            'level': level,
            'score': aggression_score,
            'confidence': 0.8
        }
    
    def analyze_conversation(self, messages: List[Dict]) -> Dict:
        """
        Analyze entire conversation for patterns and dynamics
        
        Args:
            messages: List of message dictionaries with 'sender' and 'text' keys
            
        Returns:
            Dictionary with conversation analysis
        """
        if not self.configured:
            return {
                'success': False,
                'message': 'Free AI Analyzer not configured'
            }
        
        if not messages:
            return {'success': False, 'message': 'No messages to analyze'}
        
        try:
            # Analyze each message
            message_analyses = []
            for msg in messages:
                analysis = self.analyze_message(msg.get('text', ''))
                if analysis['success']:
                    analysis['sender'] = msg.get('sender', 'unknown')
                    message_analyses.append(analysis)
            
            # Aggregate analysis
            conversation_analysis = self._aggregate_conversation_analysis(message_analyses)
            
            # Add conversation-level insights
            conversation_analysis['conversation_dynamics'] = self._analyze_conversation_dynamics(messages, message_analyses)
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'message_count': len(messages),
                'analyzed_messages': len(message_analyses),
                'analysis': conversation_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            return {
                'success': False,
                'message': f'Conversation analysis failed: {str(e)}'
            }
    
    def _aggregate_conversation_analysis(self, message_analyses: List[Dict]) -> Dict:
        """Aggregate individual message analyses into conversation-level insights"""
        if not message_analyses:
            return {}
        
        # Collect all emotions
        emotions = []
        sentiments = []
        toxicity_scores = []
        aggression_scores = []
        
        for analysis in message_analyses:
            if 'analysis' in analysis:
                data = analysis['analysis']
                
                if 'emotion' in data:
                    emotions.append(data['emotion']['primary_emotion'])
                
                if 'sentiment' in data:
                    sentiments.append(data['sentiment']['polarity'])
                
                if 'toxicity' in data:
                    toxicity_scores.append(data['toxicity']['toxicity_score'])
                
                if 'aggression' in data:
                    aggression_scores.append(data['aggression']['score'])
        
        # Calculate aggregates
        result = {}
        
        if emotions:
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            result['dominant_emotion'] = max(emotion_counts, key=emotion_counts.get)
            result['emotion_distribution'] = emotion_counts
        
        if sentiments:
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            result['overall_sentiment'] = max(sentiment_counts, key=sentiment_counts.get)
            result['sentiment_distribution'] = sentiment_counts
        
        if toxicity_scores:
            result['average_toxicity'] = sum(toxicity_scores) / len(toxicity_scores)
            result['max_toxicity'] = max(toxicity_scores)
        
        if aggression_scores:
            avg_aggression = sum(aggression_scores) / len(aggression_scores)
            result['average_aggression'] = avg_aggression
            result['aggression_level'] = 'low' if avg_aggression < 0.3 else 'medium' if avg_aggression < 0.7 else 'high'
        
        return result
    
    def _analyze_conversation_dynamics(self, messages: List[Dict], analyses: List[Dict]) -> Dict:
        """Analyze conversation dynamics and patterns"""
        if not messages or not analyses:
            return {}
        
        # Sender analysis
        senders = {}
        for i, msg in enumerate(messages):
            sender = msg.get('sender', 'unknown')
            if sender not in senders:
                senders[sender] = {
                    'message_count': 0,
                    'total_length': 0,
                    'emotions': [],
                    'toxicity_scores': [],
                    'aggression_scores': []
                }
            
            senders[sender]['message_count'] += 1
            senders[sender]['total_length'] += len(msg.get('text', ''))
            
            # Add analysis data if available
            if i < len(analyses) and analyses[i]['success']:
                analysis_data = analyses[i]['analysis']
                
                if 'emotion' in analysis_data:
                    senders[sender]['emotions'].append(analysis_data['emotion']['primary_emotion'])
                
                if 'toxicity' in analysis_data:
                    senders[sender]['toxicity_scores'].append(analysis_data['toxicity']['toxicity_score'])
                
                if 'aggression' in analysis_data:
                    senders[sender]['aggression_scores'].append(analysis_data['aggression']['score'])
        
        # Calculate sender profiles
        sender_profiles = {}
        for sender, data in senders.items():
            profile = {
                'message_count': data['message_count'],
                'avg_message_length': data['total_length'] / data['message_count'] if data['message_count'] > 0 else 0
            }
            
            if data['toxicity_scores']:
                profile['avg_toxicity'] = sum(data['toxicity_scores']) / len(data['toxicity_scores'])
            
            if data['aggression_scores']:
                profile['avg_aggression'] = sum(data['aggression_scores']) / len(data['aggression_scores'])
            
            if data['emotions']:
                emotion_counts = {}
                for emotion in data['emotions']:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                profile['dominant_emotion'] = max(emotion_counts, key=emotion_counts.get)
            
            sender_profiles[sender] = profile
        
        return {
            'sender_count': len(senders),
            'sender_profiles': sender_profiles,
            'message_distribution': {sender: data['message_count'] for sender, data in senders.items()}
        }
    
    def generate_psychological_profile(self, messages: List[Dict], forensics_data: Optional[List[Dict]] = None) -> Dict:
        """
        Generate comprehensive psychological profile
        
        Args:
            messages: List of message dictionaries
            forensics_data: Optional forensics analysis data
            
        Returns:
            Dictionary with psychological profile
        """
        if not self.configured:
            return {
                'success': False,
                'message': 'Free AI Analyzer not configured'
            }
        
        try:
            # Get conversation analysis
            conversation_analysis = self.analyze_conversation(messages)
            
            if not conversation_analysis['success']:
                return conversation_analysis
            
            analysis_data = conversation_analysis['analysis']
            
            # Generate profile based on analysis
            profile = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'profile': {
                    'personality_traits': self._extract_personality_traits(analysis_data),
                    'communication_patterns': self._extract_communication_patterns(analysis_data),
                    'emotional_regulation': self._assess_emotional_regulation(analysis_data),
                    'stress_indicators': self._identify_stress_indicators(analysis_data),
                    'relationship_dynamics': self._analyze_relationship_dynamics(analysis_data),
                    'risk_assessment': self._assess_risk_level(analysis_data),
                    'recommendations': self._generate_recommendations(analysis_data)
                }
            }
            
            # Include forensics data if available
            if forensics_data:
                profile['profile']['forensics_correlation'] = self._correlate_with_forensics(analysis_data, forensics_data)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating psychological profile: {e}")
            return {
                'success': False,
                'message': f'Profile generation failed: {str(e)}'
            }
    
    def _extract_personality_traits(self, analysis_data: Dict) -> List[str]:
        """Extract personality traits from analysis data"""
        traits = []
        
        if 'dominant_emotion' in analysis_data:
            emotion = analysis_data['dominant_emotion']
            if emotion in ['anger', 'fear']:
                traits.append('emotionally volatile')
            elif emotion in ['joy', 'happiness']:
                traits.append('generally positive')
            elif emotion == 'sadness':
                traits.append('prone to melancholy')
        
        if 'average_aggression' in analysis_data:
            if analysis_data['average_aggression'] > 0.7:
                traits.append('highly aggressive')
            elif analysis_data['average_aggression'] > 0.4:
                traits.append('moderately aggressive')
        
        if 'average_toxicity' in analysis_data:
            if analysis_data['average_toxicity'] > 0.6:
                traits.append('uses toxic language')
        
        return traits if traits else ['insufficient data for trait analysis']
    
    def _extract_communication_patterns(self, analysis_data: Dict) -> List[str]:
        """Extract communication patterns"""
        patterns = []
        
        if 'conversation_dynamics' in analysis_data:
            dynamics = analysis_data['conversation_dynamics']
            
            if 'sender_profiles' in dynamics:
                for sender, profile in dynamics['sender_profiles'].items():
                    if profile.get('avg_toxicity', 0) > 0.5:
                        patterns.append(f'{sender}: uses hostile communication')
                    
                    if profile.get('message_count', 0) > len(dynamics['sender_profiles']) * 3:
                        patterns.append(f'{sender}: dominates conversation')
        
        return patterns if patterns else ['normal communication patterns']
    
    def _assess_emotional_regulation(self, analysis_data: Dict) -> str:
        """Assess emotional regulation level"""
        if 'emotion_distribution' in analysis_data:
            emotions = analysis_data['emotion_distribution']
            negative_emotions = sum(count for emotion, count in emotions.items() 
                                  if emotion in ['anger', 'fear', 'sadness'])
            total_emotions = sum(emotions.values())
            
            if total_emotions > 0:
                negative_ratio = negative_emotions / total_emotions
                if negative_ratio > 0.7:
                    return 'low'
                elif negative_ratio > 0.4:
                    return 'moderate'
                else:
                    return 'high'
        
        return 'moderate'
    
    def _identify_stress_indicators(self, analysis_data: Dict) -> List[str]:
        """Identify stress indicators"""
        indicators = []
        
        if analysis_data.get('average_aggression', 0) > 0.5:
            indicators.append('elevated aggression levels')
        
        if analysis_data.get('average_toxicity', 0) > 0.4:
            indicators.append('increased use of hostile language')
        
        if 'dominant_emotion' in analysis_data:
            if analysis_data['dominant_emotion'] in ['anger', 'fear']:
                indicators.append('predominant negative emotions')
        
        return indicators if indicators else ['no significant stress indicators detected']
    
    def _analyze_relationship_dynamics(self, analysis_data: Dict) -> str:
        """Analyze relationship dynamics"""
        if 'conversation_dynamics' in analysis_data:
            dynamics = analysis_data['conversation_dynamics']
            sender_count = dynamics.get('sender_count', 0)
            
            if sender_count == 2:
                profiles = dynamics.get('sender_profiles', {})
                if len(profiles) == 2:
                    senders = list(profiles.keys())
                    sender1, sender2 = senders[0], senders[1]
                    
                    toxicity1 = profiles[sender1].get('avg_toxicity', 0)
                    toxicity2 = profiles[sender2].get('avg_toxicity', 0)
                    
                    if toxicity1 > 0.6 or toxicity2 > 0.6:
                        return 'hostile relationship with signs of conflict'
                    elif abs(toxicity1 - toxicity2) > 0.4:
                        return 'imbalanced relationship with one dominant party'
                    else:
                        return 'relatively balanced communication'
        
        return 'insufficient data for relationship analysis'
    
    def _assess_risk_level(self, analysis_data: Dict) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        if analysis_data.get('average_aggression', 0) > 0.7:
            risk_score += 3
        elif analysis_data.get('average_aggression', 0) > 0.4:
            risk_score += 1
        
        if analysis_data.get('average_toxicity', 0) > 0.6:
            risk_score += 2
        
        if analysis_data.get('dominant_emotion') in ['anger', 'fear']:
            risk_score += 1
        
        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis_data.get('average_aggression', 0) > 0.6:
            recommendations.append('Monitor for escalation of aggressive behavior')
        
        if analysis_data.get('average_toxicity', 0) > 0.5:
            recommendations.append('Consider intervention for toxic communication patterns')
        
        if self._assess_risk_level(analysis_data) == 'high':
            recommendations.append('High-risk situation - consider professional intervention')
        
        if not recommendations:
            recommendations.append('Continue monitoring communication patterns')
        
        return recommendations
    
    def _correlate_with_forensics(self, analysis_data: Dict, forensics_data: List[Dict]) -> Dict:
        """Correlate text analysis with forensics data"""
        # This is a placeholder for forensics correlation
        # In a real implementation, you would correlate text patterns with audio forensics
        return {
            'correlation_available': True,
            'note': 'Forensics correlation requires additional implementation'
        }


# Global instance
free_ai_analyzer = FreeAIAnalyzer()


def get_free_ai_analyzer() -> FreeAIAnalyzer:
    """Get the global free AI analyzer instance"""
    return free_ai_analyzer

