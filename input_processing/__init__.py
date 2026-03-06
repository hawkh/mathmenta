"""
Input processing package initialization.
Provides OCR and ASR capabilities for multimodal input.
"""
from .ocr import OCRProcessor, get_ocr_processor
from .asr import ASRProcessor, get_asr_processor

__all__ = ['OCRProcessor', 'get_ocr_processor', 'ASRProcessor', 'get_asr_processor']
