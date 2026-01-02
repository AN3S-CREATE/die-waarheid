import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

# Global variables for models
_gemini_model = None

def _get_gemini_model():
    """Get Gemini model lazily"""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                _gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Successfully initialized Gemini model")
            else:
                logger.warning("GEMINI_API_KEY not found")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            _gemini_model = None
    return _gemini_model

class ConversationAnalyzer:
    """Analyzes conversation data using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the conversation analyzer.
        
        Args:
            api_key: Gemini API key. If not provided, will use GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None  # Will be loaded lazily
        
        if self.api_key:
            # Store the API key for later use
            os.environ["GEMINI_API_KEY"] = self.api_key
    
    def analyze(self, messages: pd.DataFrame) -> str:
        """
        Analyze conversation data using Gemini AI.
        
        Args:
            messages: DataFrame containing conversation data
            
        Returns:
            Dictionary containing analysis results
        """
        # Get the model lazily
        self.model = _get_gemini_model()
        
        if not self.model:
            raise ValueError("Gemini model not initialized. Please provide a valid API key.")
        
        try:
            # Prepare the conversation data for analysis
            conversation_summary = self._prepare_conversation_summary(messages)
            
            # Generate the analysis prompt
            prompt = self._create_analysis_prompt(conversation_summary)
            
            # Get the analysis from Gemini
            response = self.model.generate_content(prompt)
            
            # Return the response text
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}", exc_info=True)
            return f"Error analyzing conversation: {str(e)}"
    
    def _prepare_conversation_summary(self, messages: pd.DataFrame) -> Dict:
        """Prepare a summary of the conversation for analysis."""
        if messages.empty:
            return {}
        
        # Basic conversation stats
        total_messages = len(messages)
        unique_senders = messages['sender'].nunique()
        
        # Message type distribution
        message_types = messages['message_type'].value_counts().to_dict()
        
        # Sender statistics
        sender_stats = messages['sender'].value_counts().to_dict()
        
        # Conversation duration
        time_span = (messages['timestamp'].max() - messages['timestamp'].min()).days
        
        # Prepare message samples (first and last few messages)
        sample_size = min(5, len(messages) // 2)
        early_messages = messages.head(sample_size).to_dict('records')
        recent_messages = messages.tail(sample_size).to_dict('records')
        
        # Audio analysis summary (if available)
        audio_analysis = {}
        if 'pitch_volatility' in messages.columns and 'silence_ratio' in messages.columns:
            audio_messages = messages[messages['message_type'] == 'audio'].dropna(subset=['pitch_volatility', 'silence_ratio'])
            if not audio_messages.empty:
                audio_analysis = {
                    "count": len(audio_messages),
                    "avg_pitch_volatility": audio_messages['pitch_volatility'].mean(),
                    "avg_silence_ratio": audio_messages['silence_ratio'].mean(),
                    "high_stress_messages": len(audio_messages[audio_messages['pitch_volatility'] > 50]),
                    "high_hesitation_messages": len(audio_messages[audio_messages['silence_ratio'] > 0.4])
                }
        
        return {
            "total_messages": total_messages,
            "unique_senders": unique_senders,
            "time_span_days": time_span,
            "message_types": message_types,
            "sender_stats": sender_stats,
            "early_messages": early_messages,
            "recent_messages": recent_messages,
            "audio_analysis": audio_analysis
        }
    
    def _create_analysis_prompt(self, conversation_summary: Dict) -> str:
        """Create a prompt for the Gemini model to analyze the conversation."""
        prompt = """You are an expert forensic psychologist analyzing a WhatsApp conversation. Your task is to provide a detailed analysis of the communication patterns, potential red flags, and psychological insights.

CONVERSATION SUMMARY:
{summary}

ANALYSIS INSTRUCTIONS:
1. Communication Patterns:
   - Identify the main topics of conversation
   - Note any changes in communication style over time
   - Analyze the balance of the conversation between participants

2. Emotional Tone:
   - Assess the overall emotional tone (positive, negative, neutral)
   - Identify any emotional manipulation or gaslighting attempts
   - Note any signs of stress, anger, or deception

3. Red Flags:
   - Inconsistencies in stories or statements
   - Signs of manipulation or coercion
   - Unusual patterns of communication
   - Any concerning behaviors or statements

4. Psychological Insights:
   - Power dynamics between participants
   - Potential personality traits or disorders
   - Any signs of abusive or toxic behavior

5. Recommendations:
   - Suggestions for addressing any concerns
   - When to seek professional help
   - How to improve communication

Please provide a detailed analysis in a structured JSON format with the following sections: communication_patterns, emotional_tone, red_flags, psychological_insights, and recommendations.
"""
        return prompt.format(summary=json.dumps(conversation_summary, indent=2, default=str))
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the analysis response from the model."""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, return the raw text
                return {"analysis": response_text}
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"analysis": response_text}

# Global instance for easy importing
conversation_analyzer = ConversationAnalyzer()

def analyze_conversation(messages: pd.DataFrame, api_key: Optional[str] = None) -> str:
    """
    Analyze a conversation using Gemini AI.
    
    Args:
        messages: DataFrame containing conversation data
        api_key: Optional API key (will use environment variable if not provided)
        
    Returns:
        Dictionary containing analysis results
    """
    if api_key:
        analyzer = ConversationAnalyzer(api_key=api_key)
    else:
        analyzer = conversation_analyzer
    
    return analyzer.analyze(messages)
