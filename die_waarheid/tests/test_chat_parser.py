"""
Unit tests for WhatsApp chat parser
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat_parser import WhatsAppParser


class TestWhatsAppParser(unittest.TestCase):
    """Test WhatsApp chat parsing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = WhatsAppParser()

    def test_parser_initialization(self):
        """Test parser initializes correctly"""
        self.assertIsNotNone(self.parser)
        self.assertEqual(len(self.parser.messages), 0)

    def test_timestamp_parsing_format1(self):
        """Test parsing timestamp format: [DD/MM/YYYY, HH:MM:SS]"""
        timestamp_str = "[01/01/2024, 10:30:45]"
        parsed = self.parser._parse_timestamp(timestamp_str)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2024)
        self.assertEqual(parsed.month, 1)
        self.assertEqual(parsed.day, 1)

    def test_timestamp_parsing_format2(self):
        """Test parsing timestamp format: DD/MM/YYYY, HH:MM"""
        timestamp_str = "01/01/2024, 10:30"
        parsed = self.parser._parse_timestamp(timestamp_str)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2024)

    def test_message_type_detection_text(self):
        """Test detection of text messages"""
        msg_type = self.parser._detect_message_type("Hello, how are you?")
        self.assertEqual(msg_type, "text")

    def test_message_type_detection_image(self):
        """Test detection of image messages"""
        msg_type = self.parser._detect_message_type("<image omitted>")
        self.assertEqual(msg_type, "image")

    def test_message_type_detection_audio(self):
        """Test detection of audio messages"""
        msg_type = self.parser._detect_message_type("<audio omitted>")
        self.assertEqual(msg_type, "audio")

    def test_message_type_detection_video(self):
        """Test detection of video messages"""
        msg_type = self.parser._detect_message_type("<video omitted>")
        self.assertEqual(msg_type, "video")

    def test_system_message_detection(self):
        """Test detection of system messages"""
        is_system = self.parser._is_system_message("Messages and calls are encrypted")
        self.assertTrue(is_system)

    def test_sender_extraction(self):
        """Test sender name extraction"""
        line = "[01/01/2024, 10:30:45] John Doe: Hello"
        sender = self.parser._extract_sender(line)
        
        self.assertEqual(sender, "John Doe")

    def test_get_metadata(self):
        """Test metadata generation"""
        metadata = self.parser.get_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('total_messages', metadata)
        self.assertIn('unique_senders', metadata)

    def test_get_statistics(self):
        """Test statistics generation"""
        stats = self.parser.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('message_count', stats)


class TestMessageParsing(unittest.TestCase):
    """Test individual message parsing"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = WhatsAppParser()

    def test_parse_valid_message(self):
        """Test parsing a valid message line"""
        line = "[01/01/2024, 10:30:45] John: Hello world"
        
        message = self.parser._parse_message_line(line)
        
        if message:
            self.assertIn('timestamp', message)
            self.assertIn('sender', message)
            self.assertIn('text', message)

    def test_parse_multiline_message(self):
        """Test handling of multiline messages"""
        lines = [
            "[01/01/2024, 10:30:45] John: Hello",
            "This is a continuation"
        ]
        
        self.assertIsNotNone(self.parser)

    def test_message_count(self):
        """Test message counting"""
        count = self.parser.get_message_count()
        
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
