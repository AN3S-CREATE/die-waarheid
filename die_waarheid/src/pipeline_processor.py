"""
Pipeline Processor for Die Waarheid
Automated pipeline: Upload → Transcribe → Analyze → AI Interpret → Results Table
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

from src.whisper_transcriber import WhisperTranscriber
from src.forensics import ForensicsEngine
from src.speaker_identification import SpeakerIdentificationSystem
from src.ai_analyzer import AIAnalyzer

logger = logging.getLogger(__name__)


class PipelineProcessor:
    """
    Automated forensic analysis pipeline
    Processes voice notes through complete analysis chain
    """
    
    def __init__(self, case_id: str = "MAIN_CASE"):
        self.case_id = case_id
        self.transcriber = None
        self.forensics = ForensicsEngine(use_cache=True)
        self.speaker_system = SpeakerIdentificationSystem(case_id)
        self.ai_analyzer = AIAnalyzer()
        self.results = []
        
    def initialize_transcriber(self, model_size: str = "small"):
        """Initialize Whisper transcriber with specified model"""
        if self.transcriber is None or self.transcriber.model_size != model_size:
            logger.info(f"Loading Whisper {model_size} model...")
            self.transcriber = WhisperTranscriber(model_size)
    
    def process_voice_note(
        self,
        audio_path: Path,
        language: str = "af",
        model_size: str = "small"
    ) -> Dict:
        """
        Process single voice note through complete pipeline
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            model_size: Whisper model size
            
        Returns:
            Complete analysis results dictionary
        """
        logger.info(f"Processing {audio_path.name} through pipeline...")
        
        result = {
            'filename': audio_path.name,
            'timestamp': datetime.now(),
            'success': True,
            'errors': []
        }
        
        try:
            # Step 1: Transcription
            self.initialize_transcriber(model_size)
            transcription = self.transcriber.transcribe(audio_path, language=language)
            
            if transcription.get('success'):
                result['transcription'] = transcription.get('text', '')
                result['language'] = transcription.get('language', language)
                result['duration'] = transcription.get('duration', 0)
            else:
                result['errors'].append(f"Transcription failed: {transcription.get('error')}")
                result['transcription'] = ''
            
            # Step 2: Forensic Analysis
            forensic_result = self.forensics.analyze(audio_path)
            
            if forensic_result.get('success'):
                result['stress_level'] = forensic_result.get('stress_level', 0)
                result['pitch_volatility'] = forensic_result.get('pitch_volatility', 0)
                result['silence_ratio'] = forensic_result.get('silence_ratio', 0)
                result['intensity'] = forensic_result.get('intensity', {})
                result['spectral_centroid'] = forensic_result.get('spectral_centroid', 0)
                result['audio_quality'] = forensic_result.get('audio_quality', 'unknown')
            else:
                result['errors'].append(f"Forensic analysis failed: {forensic_result.get('error')}")
            
            # Step 3: Deception Detection (based on forensics)
            result['deception_indicators'] = self._detect_deception(
                result.get('stress_level', 0),
                result.get('pitch_volatility', 0),
                result.get('silence_ratio', 0),
                result.get('transcription', '')
            )
            
            # Step 4: AI Analysis (if transcription available)
            if result.get('transcription'):
                ai_analysis = self.ai_analyzer.analyze_message(result['transcription'])
                
                if ai_analysis.get('success'):
                    result['ai_interpretation'] = ai_analysis.get('analysis', '')
                    result['sentiment'] = ai_analysis.get('sentiment', 'neutral')
                    result['gaslighting'] = self.ai_analyzer.detect_gaslighting(result['transcription'])
                    result['toxicity'] = self.ai_analyzer.detect_toxicity(result['transcription'])
                    result['narcissism'] = self.ai_analyzer.detect_narcissistic_patterns(result['transcription'])
                else:
                    result['ai_interpretation'] = "AI analysis unavailable"
                    result['errors'].append(f"AI analysis failed: {ai_analysis.get('message')}")
            else:
                result['ai_interpretation'] = "No transcription available for AI analysis"
            
            # Step 5: Speaker Identification (if system initialized)
            try:
                participants = self.speaker_system.get_all_participants()
                if len(participants) >= 2:
                    speaker_id = self.speaker_system.identify_speaker(audio_path)
                    result['identified_speaker'] = speaker_id
                else:
                    result['identified_speaker'] = "Unknown (speakers not trained)"
            except Exception as e:
                result['identified_speaker'] = f"Error: {str(e)}"
                result['errors'].append(f"Speaker identification failed: {str(e)}")
            
            # Calculate overall risk score
            result['risk_score'] = self._calculate_risk_score(result)
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for {audio_path.name}: {str(e)}")
            result['success'] = False
            result['errors'].append(f"Pipeline error: {str(e)}")
        
        self.results.append(result)
        return result
    
    def _detect_deception(
        self,
        stress_level: float,
        pitch_volatility: float,
        silence_ratio: float,
        transcription: str
    ) -> List[str]:
        """Detect deception indicators from forensic data"""
        indicators = []
        
        if stress_level > 70:
            indicators.append(f"High vocal stress ({stress_level:.1f}%)")
        
        if pitch_volatility > 50:
            indicators.append(f"Unusual pitch variation ({pitch_volatility:.1f})")
        
        if silence_ratio > 0.4:
            indicators.append(f"Excessive pauses ({silence_ratio*100:.1f}%)")
        
        # Linguistic indicators
        text_lower = transcription.lower()
        qualifier_phrases = ['honestly', 'to be honest', 'believe me', 'trust me', 'i swear']
        
        for phrase in qualifier_phrases:
            if phrase in text_lower:
                indicators.append(f"Qualifier language: '{phrase}'")
        
        return indicators
    
    def _calculate_risk_score(self, result: Dict) -> int:
        """Calculate overall risk score (0-100)"""
        score = 0
        
        # Stress contribution (0-30 points)
        stress = result.get('stress_level', 0)
        score += min(30, stress * 0.3)
        
        # Deception indicators (0-30 points)
        deception_count = len(result.get('deception_indicators', []))
        score += min(30, deception_count * 10)
        
        # Gaslighting/toxicity (0-20 points)
        if result.get('gaslighting', {}).get('gaslighting_detected'):
            score += 10
        if result.get('toxicity', {}).get('toxicity_detected'):
            score += 10
        
        # Narcissism (0-20 points)
        if result.get('narcissism', {}).get('narcissistic_patterns_detected'):
            score += 20
        
        return min(100, int(score))
    
    def process_batch(
        self,
        audio_files: List[Path],
        language: str = "af",
        model_size: str = "small"
    ) -> List[Dict]:
        """
        Process multiple voice notes through pipeline
        
        Args:
            audio_files: List of audio file paths
            language: Language code
            model_size: Whisper model size
            
        Returns:
            List of analysis results
        """
        logger.info(f"Processing batch of {len(audio_files)} files...")
        
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_path.name}")
            result = self.process_voice_note(audio_path, language, model_size)
            results.append(result)
        
        return results
    
    def get_chronological_results(self) -> List[Dict]:
        """Get all results sorted chronologically"""
        return sorted(self.results, key=lambda x: x['timestamp'])
    
    def export_results(self, output_path: Path):
        """Export results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_chronological_results(), f, indent=2, default=str)
        logger.info(f"Results exported to {output_path}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all processed files"""
        if not self.results:
            return {}
        
        total = len(self.results)
        high_risk = sum(1 for r in self.results if r.get('risk_score', 0) > 70)
        deception_detected = sum(1 for r in self.results if r.get('deception_indicators'))
        gaslighting = sum(1 for r in self.results if r.get('gaslighting', {}).get('gaslighting_detected'))
        
        avg_stress = sum(r.get('stress_level', 0) for r in self.results) / total
        avg_risk = sum(r.get('risk_score', 0) for r in self.results) / total
        
        return {
            'total_files': total,
            'high_risk_count': high_risk,
            'deception_count': deception_detected,
            'gaslighting_count': gaslighting,
            'avg_stress_level': avg_stress,
            'avg_risk_score': avg_risk
        }
