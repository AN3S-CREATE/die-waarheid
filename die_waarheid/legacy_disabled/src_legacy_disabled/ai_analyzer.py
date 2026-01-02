"""
AI Analyzer Module

Psychological profiling and forensic analysis using Google Gemini AI
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

from config import GEMINI_API_KEY, REPORTS_DIR, TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """AI-powered forensic analysis using Gemini"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = None
        if genai:
            self._initialize_model()
        
        # Forensic analysis prompts
        self.prompts = {
            'psychological_profile': """
            Analyze the following WhatsApp conversation for forensic psychological insights.
            Focus on:
            1. Emotional patterns and mood swings
            2. Communication styles (assertive, passive, aggressive)
            3. Signs of manipulation or gaslighting
            4. Attachment styles and relationship dynamics
            5. Red flags for abusive behavior
            6. Deception indicators in language patterns
            
            Conversation data:
            {conversation}
            
            Provide a structured forensic analysis with specific evidence and timestamps.
            """,
            
            'toxicity_analysis': """
            Perform a toxicity analysis on this WhatsApp conversation.
            Identify:
            1. Gaslighting techniques and frequency
            2. Narcissistic patterns (love bombing, devaluation, discard)
            3. Emotional abuse indicators
            4. Silent treatment patterns
            5. Blame-shifting and deflection tactics
            6. Controlling behaviors
            
            Highlight specific messages and patterns with timestamps.
            
            Conversation data:
            {conversation}
            """,
            
            'contradiction_analysis': """
            Analyze this WhatsApp conversation for contradictions and inconsistencies.
            Look for:
            1. Claims that contradict known facts
            2. Changes in story over time
            3. Statements that don't match behavioral patterns
            4. Promises made but not kept
            5. Lies or deceptive statements
            
            Mark each contradiction with severity (Minor/Moderate/Severe).
            
            Conversation data:
            {conversation}
            """,
            
            'risk_assessment': """
            Based on this WhatsApp conversation, provide a risk assessment.
            Evaluate:
            1. Risk level for emotional/psychological harm (Low/Medium/High/Critical)
            2. Likelihood of escalation
            3. Protective factors present
            4. Recommended interventions
            5. Urgency of action needed
            
            Consider the full context and patterns of behavior.
            
            Conversation data:
            {conversation}
            """
        }
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            if not self.api_key:
                raise ValueError("Gemini API key not provided")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Successfully initialized Gemini model")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def analyze_conversation(self, df: pd.DataFrame, analysis_type: str = 'comprehensive') -> Dict:
        """Perform AI analysis on conversation data"""
        if df.empty:
            return {'error': 'No conversation data provided'}
        
        # Prepare conversation summary
        conversation_text = self._prepare_conversation_text(df)
        
        results = {}
        
        if analysis_type == 'comprehensive':
            # Run all analysis types
            for analysis_name, prompt in self.prompts.items():
                try:
                    result = self._run_analysis(conversation_text, prompt)
                    results[analysis_name] = result
                except Exception as e:
                    logger.error(f"Failed {analysis_name} analysis: {e}")
                    results[analysis_name] = {'error': str(e)}
        else:
            # Run specific analysis
            if analysis_type in self.prompts:
                try:
                    result = self._run_analysis(conversation_text, self.prompts[analysis_type])
                    results[analysis_type] = result
                except Exception as e:
                    logger.error(f"Failed {analysis_type} analysis: {e}")
                    results[analysis_type] = {'error': str(e)}
        
        # Add quantitative metrics
        results['quantitative_metrics'] = self._calculate_quantitative_metrics(df)
        
        # Generate timestamp
        results['analysis_timestamp'] = datetime.now().strftime(TIMESTAMP_FORMAT)
        
        return results
    
    def _prepare_conversation_text(self, df: pd.DataFrame, max_messages: int = 100) -> str:
        """Prepare conversation text for AI analysis"""
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Limit messages to avoid token limits
        if len(df_sorted) > max_messages:
            # Sample evenly across the timeline
            step = len(df_sorted) // max_messages
            df_sampled = df_sorted.iloc[::step][:max_messages]
        else:
            df_sampled = df_sorted
        
        # Format messages
        messages = []
        for _, row in df_sampled.iterrows():
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            sender = row.get('sender', 'Unknown')
            message = row.get('message', '')
            
            # Add context for media messages
            if row.get('has_media'):
                message = f"[{row.get('message_type', 'media')}] {message}"
            
            messages.append(f"[{timestamp}] {sender}: {message}")
        
        return '\n'.join(messages)
    
    def _run_analysis(self, conversation_text: str, prompt: str) -> Dict:
        """Run a single analysis with Gemini"""
        try:
            full_prompt = prompt.format(conversation=conversation_text)
            
            response = self.model.generate_content(full_prompt)
            
            # Parse response
            result = {
                'analysis': response.text,
                'word_count': len(response.text.split()),
                'model_used': 'gemini-1.5-pro'
            }
            
            # Try to extract structured data
            result['structured_data'] = self._extract_structured_data(response.text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            return {'error': str(e)}
    
    def _extract_structured_data(self, text: str) -> Dict:
        """Extract structured data from AI response"""
        structured = {}
        
        # Extract patterns with regex
        patterns = {
            'risk_level': r'(?i)risk level[:\s]*(low|medium|high|critical)',
            'severity_ratings': r'(?i)(minor|moderate|severe)',
            'red_flags': r'(?i)red flag[s]?:\s*(.+?)(?=\n|$)',
            'recommendations': r'(?i)recommendation[s]?:\s*(.+?)(?=\n|$)',
            'contradictions': r'(?i)contradiction[:\s]*(.+?)(?=\n|$)'
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                structured[key] = matches
        
        return structured
    
    def _calculate_quantitative_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate quantitative metrics from conversation data"""
        metrics = {}
        
        # Basic stats
        metrics['total_messages'] = len(df)
        metrics['unique_senders'] = df['sender'].nunique()
        metrics['date_range_days'] = (df['timestamp'].max() - df['timestamp'].min()).days
        
        # Message frequency
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()
        metrics['avg_messages_per_day'] = daily_counts.mean()
        metrics['max_messages_per_day'] = daily_counts.max()
        
        # Response times (if possible)
        metrics['avg_response_time_hours'] = self._calculate_response_times(df)
        
        # Sentiment analysis
        metrics['sentiment_metrics'] = self._analyze_sentiment(df)
        
        # Media usage
        if 'has_media' in df.columns:
            metrics['media_usage'] = {
                'total_media_messages': df['has_media'].sum(),
                'media_percentage': (df['has_media'].sum() / len(df)) * 100
            }
        
        # Activity patterns
        metrics['activity_patterns'] = self._analyze_activity_patterns(df)
        
        return metrics
    
    def _calculate_response_times(self, df: pd.DataFrame) -> float:
        """Calculate average response time in hours"""
        try:
            df_sorted = df.sort_values('timestamp')
            response_times = []
            
            for i in range(1, len(df_sorted)):
                current_sender = df_sorted.iloc[i]['sender']
                prev_sender = df_sorted.iloc[i-1]['sender']
                
                if current_sender != prev_sender:
                    time_diff = df_sorted.iloc[i]['timestamp'] - df_sorted.iloc[i-1]['timestamp']
                    if time_diff.total_seconds() < 24 * 3600:  # Less than 24 hours
                        response_times.append(time_diff.total_seconds() / 3600)
            
            return sum(response_times) / len(response_times) if response_times else 0
            
        except Exception:
            return 0
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """Analyze sentiment in messages"""
        sentiments = []
        
        for message in df['message'].dropna():
            if message and not message.startswith('['):
                try:
                    blob = TextBlob(message)
                    sentiments.append(blob.sentiment.polarity)
                except:
                    pass
        
        if sentiments:
            return {
                'average_sentiment': sum(sentiments) / len(sentiments),
                'sentiment_range': (min(sentiments), max(sentiments)),
                'positive_messages': sum(1 for s in sentiments if s > 0.1),
                'negative_messages': sum(1 for s in sentiments if s < -0.1),
                'neutral_messages': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
            }
        
        return {}
    
    def _analyze_activity_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze activity patterns"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        return {
            'most_active_hour': df['hour'].mode().iloc[0] if not df['hour'].mode().empty else None,
            'most_active_day': df['day_of_week'].mode().iloc[0] if not df['day_of_week'].mode().empty else None,
            'late_night_messages': len(df[df['hour'] >= 22]),
            'early_morning_messages': len(df[df['hour'] <= 5])
        }
    
    def generate_forensic_report(self, analysis_results: Dict, output_format: str = 'markdown') -> str:
        """Generate a comprehensive forensic report"""
        report = []
        
        # Header
        report.append("# 🔬 DIE WAARHEID FORENSIC REPORT")
        report.append(f"Generated: {analysis_results.get('analysis_timestamp', 'Unknown')}")
        report.append("")
        
        # Executive Summary
        report.append("## 📊 EXECUTIVE SUMMARY")
        if 'quantitative_metrics' in analysis_results:
            metrics = analysis_results['quantitative_metrics']
            report.append(f"- Total Messages: {metrics.get('total_messages', 0)}")
            report.append(f"- Participants: {metrics.get('unique_senders', 0)}")
            report.append(f"- Analysis Period: {metrics.get('date_range_days', 0)} days")
            report.append(f"- Average Daily Messages: {metrics.get('avg_messages_per_day', 0):.1f}")
        report.append("")
        
        # Psychological Profile
        if 'psychological_profile' in analysis_results:
            report.append("## 🧠 PSYCHOLOGICAL PROFILE")
            profile = analysis_results['psychological_profile']
            if 'analysis' in profile:
                report.append(profile['analysis'])
            report.append("")
        
        # Toxicity Analysis
        if 'toxicity_analysis' in analysis_results:
            report.append("## ⚠️ TOXICITY ANALYSIS")
            toxicity = analysis_results['toxicity_analysis']
            if 'analysis' in toxicity:
                report.append(toxicity['analysis'])
            report.append("")
        
        # Contradictions
        if 'contradiction_analysis' in analysis_results:
            report.append("## 🔍 CONTRADICTIONS DETECTED")
            contradictions = analysis_results['contradiction_analysis']
            if 'analysis' in contradictions:
                report.append(contradictions['analysis'])
            report.append("")
        
        # Risk Assessment
        if 'risk_assessment' in analysis_results:
            report.append("## 🚨 RISK ASSESSMENT")
            risk = analysis_results['risk_assessment']
            if 'analysis' in risk:
                report.append(risk['analysis'])
            report.append("")
        
        # Quantitative Metrics
        if 'quantitative_metrics' in analysis_results:
            report.append("## 📈 QUANTITATIVE METRICS")
            metrics = analysis_results['quantitative_metrics']
            
            if 'sentiment_metrics' in metrics:
                report.append("### Sentiment Analysis")
                sent = metrics['sentiment_metrics']
                report.append(f"- Average Sentiment: {sent.get('average_sentiment', 0):.2f}")
                report.append(f"- Positive Messages: {sent.get('positive_messages', 0)}")
                report.append(f"- Negative Messages: {sent.get('negative_messages', 0)}")
                report.append("")
            
            if 'activity_patterns' in metrics:
                report.append("### Activity Patterns")
                activity = metrics['activity_patterns']
                report.append(f"- Most Active Hour: {activity.get('most_active_hour', 'Unknown')}:00")
                report.append(f"- Most Active Day: {activity.get('most_active_day', 'Unknown')}")
                report.append(f"- Late Night Messages (22:00+): {activity.get('late_night_messages', 0)}")
                report.append("")
        
        # Footer
        report.append("---")
        report.append("*Report generated by Die Waarheid Forensic Analysis System*")
        
        return '\n'.join(report)
    
    def save_report(self, report: str, filename: str = None) -> Path:
        """Save forensic report to file"""
        if filename is None:
            filename = f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_path = REPORTS_DIR / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report_path

# Global instance for easy importing
ai_analyzer = AIAnalyzer()
