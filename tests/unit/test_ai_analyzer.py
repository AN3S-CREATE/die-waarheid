"""
Unit tests for AIAnalyzer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ai_analyzer import AIAnalyzer


class TestAIAnalyzer:
    """Test cases for AIAnalyzer class"""

    def test_init(self):
        """Test AIAnalyzer initialization"""
        analyzer = AIAnalyzer(cache_size=100)
        assert analyzer.cache_size == 100
        assert analyzer.cache_hits == 0
        assert analyzer.cache_misses == 0
        assert hasattr(analyzer, '_cached_analyze')

    def test_sanitize_input(self):
        """Test input sanitization"""
        analyzer = AIAnalyzer()
        
        # Test normal text
        result = analyzer.sanitize_input("Hello world")
        assert result == "Hello world"
        
        # Test text with code blocks
        result = analyzer.sanitize_input("Hello ```code``` world")
        assert "```" not in result
        
        # Test max length truncation
        long_text = "a" * 1000
        result = analyzer.sanitize_input(long_text, max_length=100)
        assert len(result) == 100
        
        # Test non-string input
        result = analyzer.sanitize_input(123)
        assert result == ""

    def test_get_text_hash(self):
        """Test text hash generation"""
        analyzer = AIAnalyzer()
        
        hash1 = analyzer._get_text_hash("Hello world")
        hash2 = analyzer._get_text_hash("Hello world")
        hash3 = analyzer._get_text_hash("Different text")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hash length

    @patch('src.ai_analyzer.genai')
    def test_analyze_uncached_success(self, mock_genai):
        """Test successful uncached analysis"""
        # Setup mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"emotion": "happy", "toxicity_score": 0.1, "aggression_level": "low", "confidence": 0.9}'
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        analyzer = AIAnalyzer()
        analyzer.configured = True
        analyzer.model = mock_model
        
        result = analyzer._analyze_uncached("test_hash", "I am happy today!")
        
        assert result['success']
        assert result['emotion'] == 'happy'
        assert result['toxicity_score'] == 0.1
        assert result['aggression_level'] == 'low'
        assert result['confidence'] == 0.9
        assert not result['cached']

    @patch('src.ai_analyzer.genai')
    def test_analyze_uncached_quota_exceeded(self, mock_genai):
        """Test handling of quota exceeded error"""
        # Setup mock to raise quota error
        mock_genai.GenerativeModel.side_effect = Exception("429 quota exceeded")
        
        analyzer = AIAnalyzer()
        analyzer.configured = True
        
        result = analyzer._analyze_uncached("test_hash", "test message")
        
        assert not result['success']
        assert result['error_type'] == 'quota_exceeded'
        assert "quota exceeded" in result['error']

    def test_analyze_message_fallback(self):
        """Test fallback analysis"""
        analyzer = AIAnalyzer()
        
        # Test toxic message
        result = analyzer.analyze_message_fallback("You are stupid and worthless!")
        assert result['success']
        assert result['emotion'] == 'negative'
        assert result['toxicity_score'] > 0
        assert result['fallback']
        
        # Test positive message
        result = analyzer.analyze_message_fallback("I love you so much!")
        assert result['emotion'] == 'positive'
        assert result['toxicity_score'] == 0
        
        # Test aggressive message
        result = analyzer.analyze_message_fallback("THIS IS TERRIBLE!!!")
        assert result['aggression_level'] == 'high'

    def test_detect_gaslighting(self):
        """Test gaslighting detection"""
        analyzer = AIAnalyzer()
        
        # Test with gaslighting phrases
        result = analyzer.detect_gaslighting("You're remembering it wrong, that never happened")
        assert result['gaslighting_detected']
        assert len(result['detected_phrases']) > 0
        
        # Test without gaslighting
        result = analyzer.detect_gaslighting("Hello, how are you today?")
        assert not result['gaslighting_detected']

    def test_detect_toxicity(self):
        """Test toxicity detection"""
        analyzer = AIAnalyzer()
        
        # Test toxic message
        result = analyzer.detect_toxicity("You're an idiot and I hate you")
        assert result['toxicity_detected']
        assert result['toxicity_score'] > 0
        
        # Test non-toxic message
        result = analyzer.detect_toxicity("Have a nice day!")
        assert not result['toxicity_detected']

    def test_detect_narcissistic_patterns(self):
        """Test narcissism detection"""
        analyzer = AIAnalyzer()
        
        # Test narcissistic message
        result = analyzer.detect_narcissistic_patterns("I'm the best and everyone should admire me")
        assert result['narcissistic_patterns_detected']
        assert result['narcissism_score'] > 0
        
        # Test normal message
        result = analyzer.detect_narcissistic_patterns("We did a great job together")
        assert not result['narcissistic_patterns_detected']

    def test_get_cache_stats(self):
        """Test cache statistics"""
        analyzer = AIAnalyzer()
        
        # Initially empty
        stats = analyzer.get_cache_stats()
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['hit_rate_percent'] == 0
        
        # Simulate some hits and misses
        analyzer.cache_hits = 8
        analyzer.cache_misses = 2
        
        stats = analyzer.get_cache_stats()
        assert stats['cache_hits'] == 8
        assert stats['cache_misses'] == 2
        assert stats['hit_rate_percent'] == 80.0

    def test_clear_cache(self):
        """Test cache clearing"""
        analyzer = AIAnalyzer()
        
        # Add some stats
        analyzer.cache_hits = 10
        analyzer.cache_misses = 5
        
        # Clear cache
        analyzer.clear_cache()
        
        # Verify reset
        assert analyzer.cache_hits == 0
        assert analyzer.cache_misses == 0

    @patch.object(AIAnalyzer, '_analyze_uncached')
    def test_analyze_message_with_cache(self, mock_analyze):
        """Test analyze_message with caching"""
        # Setup mock to return success
        mock_analyze.return_value = {
            'success': True,
            'emotion': 'neutral',
            'cached': False
        }
        
        analyzer = AIAnalyzer()
        
        # First call - should be miss
        result1 = analyzer.analyze_message("test message")
        assert result1['success']
        assert analyzer.cache_misses == 1
        
        # Second call - should be hit
        result2 = analyzer.analyze_message("test message")
        assert result2['success']
        assert analyzer.cache_hits == 1

    @patch.object(AIAnalyzer, '_analyze_uncached')
    def test_analyze_message_fallback_on_error(self, mock_analyze):
        """Test fallback when uncached analysis fails"""
        # Setup mock to raise exception
        mock_analyze.side_effect = Exception("API error")
        
        analyzer = AIAnalyzer()
        
        result = analyzer.analyze_message("test message")
        
        # Should return fallback result
        assert result['success']
        assert result.get('fallback', False)

    def test_pattern_based_emotion_detection(self):
        """Test emotion detection in fallback analysis"""
        analyzer = AIAnalyzer()
        
        # Positive emotions
        assert analyzer.analyze_message_fallback("I am so happy and love this!")['emotion'] == 'positive'
        
        # Negative emotions
        assert analyzer.analyze_message_fallback("I hate this terrible thing")['emotion'] == 'negative'
        
        # Neutral
        assert analyzer.analyze_message_fallback("This is a statement")['emotion'] == 'neutral'

    def test_aggression_level_detection(self):
        """Test aggression level detection"""
        analyzer = AIAnalyzer()
        
        # High aggression
        result = analyzer.analyze_message_fallback("THIS IS SO TERRIBLE!!!")
        assert result['aggression_level'] == 'high'
        
        # Medium aggression
        result = analyzer.analyze_message_fallback("This is bad!")
        assert result['aggression_level'] == 'medium'
        
        # Low aggression
        result = analyzer.analyze_message_fallback("This is fine.")
        assert result['aggression_level'] == 'low'
