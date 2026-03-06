"""
Utility functions for Math Mentor.
"""
from .tools import SafeCalculator, get_calculator, safe_calculate

__all__ = [
    'format_math_text',
    'truncate_text',
    'calculate_confidence_color',
    'SafeCalculator',
    'get_calculator',
    'safe_calculate'
]


def format_math_text(text: str) -> str:
    """
    Format mathematical text for better display.
    
    Args:
        text: Raw mathematical text
        
    Returns:
        Formatted text
    """
    # Common math formatting improvements
    replacements = {
        'x²': 'x^2',
        'x³': 'x^3',
        '√': 'sqrt(',
        'π': 'pi',
        '∞': 'infinity',
        '∫': '∫ ',
        '∑': 'Σ ',
        '→': ' -> ',
        '≥': ' >= ',
        '≤': ' <= ',
        '≠': ' != ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def calculate_confidence_color(confidence: float) -> str:
    """
    Get color class for confidence level.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Color class name
    """
    if confidence >= 0.8:
        return "success"
    elif confidence >= 0.6:
        return "warning"
    else:
        return "danger"


__all__ = [
    'format_math_text',
    'truncate_text',
    'calculate_confidence_color'
]
