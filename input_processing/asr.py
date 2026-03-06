"""
ASR (Automatic Speech Recognition) module for Math Mentor.
Uses OpenAI's Whisper API for speech-to-text conversion.
"""
import os
import tempfile
from typing import Dict, Any, Optional
from config import Config


class ASRProcessor:
    """
    Process audio to extract spoken mathematical content.
    Uses OpenAI's Whisper API for accurate speech recognition.
    """
    
    def __init__(self):
        """Initialize the ASR processor."""
        self.confidence_threshold = Config.ASR_CONFIDENCE_THRESHOLD
        self._client = None
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for ASR. Set it in .env file.")
            from openai import OpenAI
            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._client
    
    def process_audio(self, audio_data: bytes, file_extension: str = "wav") -> Dict[str, Any]:
        """
        Process audio data to extract spoken content.
        
        Args:
            audio_data: Raw audio bytes
            file_extension: Audio file extension (wav, mp3, m4a, etc.)
            
        Returns:
            Dictionary with transcribed text, confidence, and metadata
        """
        try:
            client = self._get_client()
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{file_extension}",
                delete=False
            ) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Use Whisper API for transcription
                with open(temp_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                
                # Extract information
                text = transcript.text
                confidence = 0.85  # Whisper doesn't provide per-word confidence in simple API
                
                # Check for language detection
                language = getattr(transcript, 'language', 'en')
                
                # Determine if human review is needed
                # For now, use a simple heuristic based on text quality
                needs_review = self._assess_transcription_quality(text)
                
                return {
                    'success': True,
                    'extracted_text': text,
                    'confidence': confidence,
                    'notes': f"Language detected: {language}",
                    'needs_review': needs_review,
                    'input_type': 'audio',
                    'error': None
                }
                
            finally:
                # Clean up temp file
                os.unlink(temp_path)
                
        except ValueError as e:
            # OpenAI API key not configured
            return {
                'success': False,
                'extracted_text': "",
                'confidence': 0.0,
                'notes': "OpenAI API key not configured. Please add OPENAI_API_KEY to .env file.",
                'needs_review': True,
                'input_type': 'audio',
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'extracted_text': "",
                'confidence': 0.0,
                'notes': f"ASR processing failed: {str(e)}",
                'needs_review': True,
                'input_type': 'audio',
                'error': str(e)
            }
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Process an audio file to extract spoken content.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcribed text, confidence, and metadata
        """
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            file_ext = os.path.splitext(audio_path)[1][1:]  # Remove the dot
            return self.process_audio(audio_data, file_ext)
            
        except Exception as e:
            return {
                'success': False,
                'extracted_text': "",
                'confidence': 0.0,
                'notes': f"Failed to load audio file: {str(e)}",
                'needs_review': True,
                'input_type': 'audio',
                'error': str(e)
            }
    
    def _assess_transcription_quality(self, text: str) -> bool:
        """
        Assess if the transcription needs human review.
        
        Args:
            text: Transcribed text
            
        Returns:
            True if review is needed, False otherwise
        """
        # Simple heuristics for quality assessment
        text = text.strip()
        
        # Empty or very short transcription
        if len(text) < 10:
            return True
        
        # Check for common speech recognition errors
        error_patterns = [
            "[inaudible]",
            "[unintelligible]",
            "...",
        ]
        
        for pattern in error_patterns:
            if pattern in text.lower():
                return True
        
        # Check for incomplete sentences (no punctuation)
        if not any(p in text for p in '.!?'):
            return True
        
        return False


# Singleton instance
_asr_instance = None


def get_asr_processor() -> ASRProcessor:
    """Get or create the ASR processor singleton."""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASRProcessor()
    return _asr_instance
