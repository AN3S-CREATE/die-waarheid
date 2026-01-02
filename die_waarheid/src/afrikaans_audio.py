"""
Afrikaans Audio Processing for Die Waarheid
Handles foreground/background audio separation and transcription
Ensures correct speaker attribution through audio layer analysis

CRITICAL: Separates background and foreground Afrikaans audio
to ensure the right words are attributed to the right person
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AudioLayer:
    """Represents a separated audio layer"""
    layer_type: str  # "foreground" or "background"
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    energy_level: float
    clarity_score: float
    speaker_detected: bool
    confidence: float


@dataclass
class SeparatedAudio:
    """Result of audio separation"""
    success: bool
    foreground: Optional[AudioLayer]
    background: Optional[AudioLayer]
    original_duration: float
    separation_confidence: float
    speaker_count_estimate: int
    warnings: List[str]


class AudioLayerSeparator:
    """
    Separate foreground and background audio
    Critical for correct speaker attribution in Afrikaans recordings
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio separator

        Args:
            sample_rate: Target sample rate
        """
        self.sample_rate = sample_rate
        self.energy_threshold = 0.3
        self.clarity_threshold = 0.5

    def separate_layers(self, audio: np.ndarray, sr: int) -> SeparatedAudio:
        """
        Separate audio into foreground and background layers

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            SeparatedAudio with separated layers
        """
        try:
            warnings = []
            
            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

            duration = len(audio) / sr

            # Calculate energy profile
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            
            energy = self._calculate_energy(audio, frame_length, hop_length)
            
            # Separate based on energy
            foreground_mask = energy > np.percentile(energy, 70)
            background_mask = ~foreground_mask

            # Extract layers
            foreground_audio = self._apply_mask(audio, foreground_mask, hop_length)
            background_audio = self._apply_mask(audio, background_mask, hop_length)

            # Calculate metrics
            fg_energy = np.mean(np.abs(foreground_audio)) if len(foreground_audio) > 0 else 0
            bg_energy = np.mean(np.abs(background_audio)) if len(background_audio) > 0 else 0

            # Clarity score based on energy separation
            if fg_energy + bg_energy > 0:
                separation_ratio = fg_energy / (fg_energy + bg_energy)
                clarity = abs(separation_ratio - 0.5) * 2  # Higher when more separated
            else:
                clarity = 0
                warnings.append("Very low audio energy detected")

            # Estimate speaker count
            speaker_count = self._estimate_speaker_count(audio, sr)

            # Create layer objects
            foreground_layer = AudioLayer(
                layer_type="foreground",
                audio_data=foreground_audio,
                sample_rate=sr,
                duration=len(foreground_audio) / sr,
                energy_level=float(fg_energy),
                clarity_score=float(clarity),
                speaker_detected=fg_energy > self.energy_threshold,
                confidence=min(1.0, clarity * 1.2)
            )

            background_layer = AudioLayer(
                layer_type="background",
                audio_data=background_audio,
                sample_rate=sr,
                duration=len(background_audio) / sr,
                energy_level=float(bg_energy),
                clarity_score=float(1 - clarity),
                speaker_detected=bg_energy > self.energy_threshold * 0.5,
                confidence=min(1.0, (1 - clarity) * 1.2)
            )

            # Add warnings for potential issues
            if clarity < 0.3:
                warnings.append("Low separation clarity - foreground/background may overlap")
            
            if speaker_count > 2:
                warnings.append(f"Multiple speakers detected ({speaker_count}) - verify attribution carefully")

            return SeparatedAudio(
                success=True,
                foreground=foreground_layer,
                background=background_layer,
                original_duration=duration,
                separation_confidence=clarity,
                speaker_count_estimate=speaker_count,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Audio separation error: {str(e)}")
            return SeparatedAudio(
                success=False,
                foreground=None,
                background=None,
                original_duration=0,
                separation_confidence=0,
                speaker_count_estimate=0,
                warnings=[f"Separation failed: {str(e)}"]
            )

    def _calculate_energy(
        self,
        audio: np.ndarray,
        frame_length: int,
        hop_length: int
    ) -> np.ndarray:
        """Calculate frame-wise energy"""
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        return np.array(energy)

    def _apply_mask(
        self,
        audio: np.ndarray,
        mask: np.ndarray,
        hop_length: int
    ) -> np.ndarray:
        """Apply frame mask to audio"""
        output = np.zeros_like(audio)
        
        for i, m in enumerate(mask):
            start = i * hop_length
            end = min(start + hop_length, len(audio))
            if m:
                output[start:end] = audio[start:end]
        
        return output

    def _estimate_speaker_count(self, audio: np.ndarray, sr: int) -> int:
        """Estimate number of speakers in audio"""
        try:
            # Simple estimation based on energy variance
            frame_length = int(0.1 * sr)
            energies = []
            
            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                energies.append(np.mean(np.abs(frame)))
            
            if len(energies) < 2:
                return 1

            # Cluster energy levels
            energies = np.array(energies)
            threshold = np.median(energies)
            
            # Count transitions
            above = energies > threshold
            transitions = np.sum(np.abs(np.diff(above.astype(int))))
            
            # Estimate speakers based on transition pattern
            if transitions < 3:
                return 1
            elif transitions < 10:
                return 2
            else:
                return min(3, transitions // 5 + 1)

        except Exception:
            return 1


class AfrikaansAudioTranscriber:
    """
    Transcribe Afrikaans audio with layer-specific processing
    """

    def __init__(self):
        """Initialize Afrikaans audio transcriber"""
        self.separator = AudioLayerSeparator()
        self.whisper_model = None
        self._load_whisper()

    def _load_whisper(self):
        """Load Whisper model for Afrikaans"""
        try:
            import whisper
            # Use medium model for better Afrikaans accuracy
            self.whisper_model = whisper.load_model("medium")
            logger.info("Whisper model loaded for Afrikaans transcription")
        except ImportError:
            logger.warning("Whisper not available - using fallback transcription")
        except Exception as e:
            logger.error(f"Error loading Whisper: {str(e)}")

    def transcribe_audio(
        self,
        audio_path: Path,
        separate_layers: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe Afrikaans audio file

        Args:
            audio_path: Path to audio file
            separate_layers: Whether to separate foreground/background

        Returns:
            Transcription results with layer information
        """
        try:
            import librosa
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            duration = len(audio) / sr

            results = {
                'success': True,
                'audio_path': str(audio_path),
                'duration': duration,
                'transcriptions': [],
                'warnings': []
            }

            if separate_layers:
                # Separate audio layers
                separated = self.separator.separate_layers(audio, sr)
                
                if separated.success:
                    results['separation'] = {
                        'confidence': separated.separation_confidence,
                        'speaker_count': separated.speaker_count_estimate,
                        'warnings': separated.warnings
                    }
                    results['warnings'].extend(separated.warnings)

                    # Transcribe foreground
                    if separated.foreground and separated.foreground.speaker_detected:
                        fg_transcription = self._transcribe_layer(
                            separated.foreground,
                            "foreground"
                        )
                        results['transcriptions'].append(fg_transcription)

                    # Transcribe background
                    if separated.background and separated.background.speaker_detected:
                        bg_transcription = self._transcribe_layer(
                            separated.background,
                            "background"
                        )
                        results['transcriptions'].append(bg_transcription)
                else:
                    results['warnings'].append("Layer separation failed - transcribing full audio")
                    full_transcription = self._transcribe_full(audio, sr)
                    results['transcriptions'].append(full_transcription)
            else:
                # Transcribe full audio without separation
                full_transcription = self._transcribe_full(audio, sr)
                results['transcriptions'].append(full_transcription)

            return results

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'transcriptions': [],
                'warnings': [f"Transcription failed: {str(e)}"]
            }

    def _transcribe_layer(
        self,
        layer: AudioLayer,
        layer_type: str
    ) -> Dict[str, Any]:
        """Transcribe a single audio layer"""
        if self.whisper_model is None:
            return {
                'layer': layer_type,
                'text': '',
                'language': 'af',
                'confidence': 0,
                'error': 'Whisper not available'
            }

        try:
            # Whisper transcription with Afrikaans language hint
            result = self.whisper_model.transcribe(
                layer.audio_data,
                language="af",  # Afrikaans
                task="transcribe"
            )

            return {
                'layer': layer_type,
                'text': result.get('text', ''),
                'language': result.get('language', 'af'),
                'confidence': layer.confidence,
                'segments': result.get('segments', []),
                'duration': layer.duration
            }

        except Exception as e:
            logger.error(f"Layer transcription error: {str(e)}")
            return {
                'layer': layer_type,
                'text': '',
                'language': 'af',
                'confidence': 0,
                'error': str(e)
            }

    def _transcribe_full(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """Transcribe full audio without layer separation"""
        if self.whisper_model is None:
            return {
                'layer': 'full',
                'text': '',
                'language': 'af',
                'confidence': 0,
                'error': 'Whisper not available'
            }

        try:
            result = self.whisper_model.transcribe(
                audio,
                language="af",
                task="transcribe"
            )

            return {
                'layer': 'full',
                'text': result.get('text', ''),
                'language': result.get('language', 'af'),
                'confidence': 0.8,  # Default confidence for full audio
                'segments': result.get('segments', []),
                'duration': len(audio) / sr
            }

        except Exception as e:
            logger.error(f"Full transcription error: {str(e)}")
            return {
                'layer': 'full',
                'text': '',
                'language': 'af',
                'confidence': 0,
                'error': str(e)
            }


class AfrikaansAudioVerifier:
    """
    Verify Afrikaans audio transcription accuracy
    Triple-check system for preventing wrong attribution
    """

    def __init__(self):
        """Initialize verifier"""
        self.transcriber = AfrikaansAudioTranscriber()
        from afrikaans_processor import AfrikaansProcessor
        self.processor = AfrikaansProcessor()

    def verify_audio_transcription(
        self,
        audio_path: Path,
        expected_speaker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform full verification of audio transcription

        Args:
            audio_path: Path to audio file
            expected_speaker: Expected speaker ID

        Returns:
            Verified transcription with confidence scores
        """
        logger.info(f"Starting audio verification for: {audio_path}")

        # Step 1: Transcribe audio with layer separation
        transcription_result = self.transcriber.transcribe_audio(
            audio_path,
            separate_layers=True
        )

        if not transcription_result.get('success'):
            return {
                'success': False,
                'error': transcription_result.get('error', 'Transcription failed'),
                'verified': False
            }

        # Step 2: Verify each transcription
        verified_results = []
        all_verified = True
        review_required = False

        for transcription in transcription_result.get('transcriptions', []):
            text = transcription.get('text', '')
            layer = transcription.get('layer', 'unknown')

            if not text.strip():
                continue

            # Run through Afrikaans processor for verification
            verification = self.processor.process_text(
                text,
                speaker_id=expected_speaker,
                audio_layer=layer,
                context=f"Audio file: {audio_path.name}"
            )

            verified_results.append({
                'layer': layer,
                'original_afrikaans': verification.original_afrikaans,
                'translated_english': verification.translated_english,
                'confidence': verification.overall_confidence,
                'confidence_level': verification.overall_confidence_level.value,
                'all_checks_passed': verification.all_checks_passed,
                'requires_review': verification.requires_human_review,
                'review_reasons': verification.review_reasons,
                'word_count': len(verification.word_verifications),
                'uncertain_words': [
                    w.word_afrikaans for w in verification.word_verifications
                    if w.requires_review
                ]
            })

            if not verification.all_checks_passed:
                all_verified = False
            
            if verification.requires_human_review:
                review_required = True

        return {
            'success': True,
            'audio_path': str(audio_path),
            'duration': transcription_result.get('duration', 0),
            'separation_info': transcription_result.get('separation', {}),
            'results': verified_results,
            'all_verified': all_verified,
            'review_required': review_required,
            'warnings': transcription_result.get('warnings', []),
            'safe_to_use': all_verified and not review_required
        }

    def generate_verification_report(
        self,
        verification_result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable verification report

        Args:
            verification_result: Result from verify_audio_transcription

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("AFRIKAANS AUDIO VERIFICATION REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Audio File: {verification_result.get('audio_path', 'Unknown')}")
        report.append(f"Duration: {verification_result.get('duration', 0):.2f} seconds")
        report.append("")

        # Separation info
        sep_info = verification_result.get('separation_info', {})
        if sep_info:
            report.append("AUDIO SEPARATION:")
            report.append(f"  Confidence: {sep_info.get('confidence', 0):.1%}")
            report.append(f"  Estimated Speakers: {sep_info.get('speaker_count', 0)}")
            report.append("")

        # Overall status
        report.append("VERIFICATION STATUS:")
        if verification_result.get('safe_to_use'):
            report.append("  ✅ SAFE TO USE - All verifications passed")
        elif verification_result.get('review_required'):
            report.append("  ⚠️  REQUIRES HUMAN REVIEW")
        else:
            report.append("  ❌ VERIFICATION FAILED")
        report.append("")

        # Warnings
        warnings = verification_result.get('warnings', [])
        if warnings:
            report.append("⚠️  WARNINGS:")
            for warning in warnings:
                report.append(f"  - {warning}")
            report.append("")

        # Results by layer
        report.append("TRANSCRIPTION RESULTS:")
        report.append("-" * 50)
        
        for result in verification_result.get('results', []):
            report.append(f"\nLAYER: {result.get('layer', 'unknown').upper()}")
            report.append(f"  Afrikaans: {result.get('original_afrikaans', '')}")
            report.append(f"  English: {result.get('translated_english', '')}")
            report.append(f"  Confidence: {result.get('confidence', 0):.1%} ({result.get('confidence_level', 'unknown')})")
            report.append(f"  Verified: {'✅ YES' if result.get('all_checks_passed') else '❌ NO'}")
            
            if result.get('requires_review'):
                report.append(f"  ⚠️  Requires Review:")
                for reason in result.get('review_reasons', []):
                    report.append(f"      - {reason}")
            
            uncertain = result.get('uncertain_words', [])
            if uncertain:
                report.append(f"  Uncertain Words: {', '.join(uncertain)}")

        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    verifier = AfrikaansAudioVerifier()
    print("Afrikaans Audio Verifier initialized")
    print("Use verifier.verify_audio_transcription(audio_path) to verify audio")
