"""
Die Waarheid - Digital Forensic Analysis Tool

This package provides tools for analyzing WhatsApp conversations with a focus on detecting
deception, stress, and other psychological markers in both text and audio messages.
"""

try:
    from .forensics import analyze_audio_file, AudioAnalyzer
except Exception:
    analyze_audio_file = None
    AudioAnalyzer = None

try:
    from .processor import process_whatsapp_export, WhatsAppProcessor
except Exception:
    process_whatsapp_export = None
    WhatsAppProcessor = None

try:
    from .brain import analyze_conversation, ConversationAnalyzer
except Exception:
    analyze_conversation = None
    ConversationAnalyzer = None

__version__ = '1.0.0'
__all__ = [
    'analyze_audio_file',
    'AudioAnalyzer',
    'process_whatsapp_export',
    'WhatsAppProcessor',
    'analyze_conversation',
    'ConversationAnalyzer'
]
