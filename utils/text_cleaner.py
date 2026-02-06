"""
Text Cleaning Module
Pre-processes raw transcribed text for improvement
"""

import re
import string
from typing import List, Set


# Common filler words to remove
FILLER_WORDS: Set[str] = {
    'uh', 'um', 'uhm', 'erm', 'er', 'ah', 'eh',
    'like', 'you know', 'i mean', 'basically',
    'actually', 'literally', 'honestly', 'right',
    'so yeah', 'yeah', 'okay so', 'well',
}

# Repeated words patterns
REPEATED_WORDS_PATTERN = re.compile(r'\b(\w+)(\s+\1\b)+', re.IGNORECASE)


def remove_filler_words(text: str, custom_fillers: Set[str] = None) -> str:
    """
    Remove filler words from text.
    
    Args:
        text: Input text
        custom_fillers: Additional filler words to remove
        
    Returns:
        Text with filler words removed
    """
    fillers = FILLER_WORDS.copy()
    if custom_fillers:
        fillers.update(custom_fillers)
    
    # Sort by length (longest first) to handle multi-word fillers
    sorted_fillers = sorted(fillers, key=len, reverse=True)
    
    result = text
    for filler in sorted_fillers:
        # Create pattern that matches filler as whole word/phrase
        pattern = r'\b' + re.escape(filler) + r'\b'
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def remove_repeated_words(text: str) -> str:
    """
    Remove accidentally repeated words (e.g., "the the" -> "the").
    
    Args:
        text: Input text
        
    Returns:
        Text with repeated words removed
    """
    return REPEATED_WORDS_PATTERN.sub(r'\1', text)


def fix_punctuation_spacing(text: str) -> str:
    """
    Fix spacing around punctuation marks.
    
    Args:
        text: Input text
        
    Returns:
        Text with corrected punctuation spacing
    """
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Add space after punctuation if missing
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    
    # Fix multiple punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    return text


def capitalize_sentences(text: str) -> str:
    """
    Properly capitalize the beginning of sentences.
    
    Args:
        text: Input text
        
    Returns:
        Text with proper sentence capitalization
    """
    # Split by sentence endings
    sentences = re.split(r'([.!?]+\s*)', text)
    
    result = []
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:  # This is a sentence, not a separator
            # Capitalize first letter
            part = part.strip()
            if part:
                part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
        result.append(part)
    
    return ''.join(result)


def clean_text(text: str, options: dict = None) -> str:
    """
    Apply all cleaning operations to text.
    
    Args:
        text: Raw transcribed text
        options: Dictionary of cleaning options
            - remove_fillers: bool (default: True)
            - remove_repeated: bool (default: True)
            - fix_punctuation: bool (default: True)
            - capitalize: bool (default: True)
            
    Returns:
        Cleaned text
    """
    if options is None:
        options = {}
    
    # Default options
    remove_fillers = options.get('remove_fillers', True)
    remove_repeated = options.get('remove_repeated', True)
    fix_punctuation = options.get('fix_punctuation', True)
    capitalize = options.get('capitalize', True)
    
    result = text
    
    # Apply cleaning steps
    if remove_fillers:
        result = remove_filler_words(result)
    
    if remove_repeated:
        result = remove_repeated_words(result)
    
    if fix_punctuation:
        result = fix_punctuation_spacing(result)
    
    if capitalize:
        result = capitalize_sentences(result)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


if __name__ == "__main__":
    # Test the module
    test_text = "uh so like I want to um explain this new app you know and uh it's really good"
    print("Original:", test_text)
    print("Cleaned:", clean_text(test_text))
