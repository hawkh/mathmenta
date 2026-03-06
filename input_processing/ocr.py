"""
OCR (Optical Character Recognition) module for Math Mentor.
Uses Claude's vision capabilities for robust math text extraction.
"""
import base64
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import io
from config import Config


class OCRProcessor:
    """
    Process images to extract mathematical text and problems.
    Uses Claude's vision API for high accuracy on mathematical notation.
    """
    
    def __init__(self):
        """Initialize the OCR processor."""
        self.confidence_threshold = Config.OCR_CONFIDENCE_THRESHOLD
        self._client = None
    
    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        return self._client
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image to extract mathematical content.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        try:
            client = self._get_client()
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Use Claude's vision API for OCR
            response = client.messages.create(
                model=Config.DEFAULT_MODEL,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract all mathematical text from this image. 
                            
                            IMPORTANT:
                            1. Transcribe ALL text exactly as it appears
                            2. Use LaTeX formatting for mathematical expressions (e.g., $x^2$, $\frac{a}{b}$)
                            3. Preserve the structure and organization
                            4. Include any diagrams or figures descriptions in [brackets]
                            5. If there are multiple problems, separate them clearly
                            
                            Output format:
                            EXTRACTED_TEXT:
                            [Your extracted text here]
                            
                            CONFIDENCE: [0.0 to 1.0]
                            
                            NOTES: [Any ambiguities or issues]"""
                        }
                    ]
                }]
            )
            
            # Parse response
            response_text = response.content[0].text
            
            # Extract components
            extracted_text = ""
            confidence = 0.85  # Default confidence
            notes = ""
            
            if "EXTRACTED_TEXT:" in response_text:
                parts = response_text.split("EXTRACTED_TEXT:")
                if len(parts) > 1:
                    remaining = parts[1]
                    
                    if "CONFIDENCE:" in remaining:
                        text_part, conf_part = remaining.split("CONFIDENCE:")
                        extracted_text = text_part.strip()
                        
                        if "NOTES:" in conf_part:
                            conf_str, notes_part = conf_part.split("NOTES:")
                            confidence = float(conf_str.strip().replace(":", ""))
                            notes = notes_part.strip()
                        else:
                            confidence = float(conf_part.strip().replace(":", "").split()[0])
                    else:
                        extracted_text = remaining.split("NOTES:")[0].strip()
                        if "NOTES:" in response_text:
                            notes = response_text.split("NOTES:")[1].strip()
            else:
                extracted_text = response_text
            
            # Determine if human review is needed
            needs_review = confidence < self.confidence_threshold
            
            return {
                'success': True,
                'extracted_text': extracted_text,
                'confidence': confidence,
                'notes': notes,
                'needs_review': needs_review,
                'input_type': 'image',
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'extracted_text': "",
                'confidence': 0.0,
                'notes': f"OCR processing failed: {str(e)}",
                'needs_review': True,
                'input_type': 'image',
                'error': str(e)
            }
    
    def process_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image file to extract mathematical content.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        try:
            image = Image.open(image_path)
            return self.process_image(image)
        except Exception as e:
            return {
                'success': False,
                'extracted_text': "",
                'confidence': 0.0,
                'notes': f"Failed to load image: {str(e)}",
                'needs_review': True,
                'input_type': 'image',
                'error': str(e)
            }


# Singleton instance
_ocr_instance = None


def get_ocr_processor() -> OCRProcessor:
    """Get or create the OCR processor singleton."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRProcessor()
    return _ocr_instance
