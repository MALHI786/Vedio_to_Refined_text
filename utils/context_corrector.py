"""
Context-Aware Text Correction Module
Fixes transcription errors and domain-specific terminology

Key Features:
- Contextual word replacement (hair → her, tenders → attendance)
- Domain-specific terminology standardization
- Transcription artifact removal
- Phonetic similarity-based corrections
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class CorrectionRule:
    """A single correction rule with context."""
    wrong: str
    correct: str
    context_keywords: List[str]  # Context that triggers this correction
    priority: int = 1  # Higher = more important


# ============================================================================
# TRANSCRIPTION ERROR MAPPINGS
# Common speech-to-text mistakes (phonetically similar words)
# ============================================================================
TRANSCRIPTION_CORRECTIONS: Dict[str, str] = {
    # Phonetic confusions - Critical Fixes
    "epsilon": "absent",
    "resisting": "registration",
    "resisted": "registered",
    "resist": "register",
    "attenders": "attendance",
    "attender": "attendance",
    
    # Hair → Her (very common)
    "hair": "her",
    "hare": "her",
    "send hair": "send her",
    "to hair": "to her",
    "tell hair": "tell her",
    
    # Common word confusions
    "there": "their",  # Context-dependent
    "your": "you're",  # Context-dependent
    "its": "it's",     # Context-dependent
    "four": "for",     # Context-dependent
    "won": "one",      # Context-dependent
    "sum": "some",     # Context-dependent
    "wright": "write",
    "right": "write",  # Context-dependent in tech
    "rite": "write",
    "no": "know",      # Context-dependent
    "sea": "see",      # Context-dependent
    "too": "to",       # Context-dependent
    "two": "to",       # Context-dependent
    "buy": "by",       # Context-dependent
    "bye": "by",       # Context-dependent
    "our": "hour",     # Context-dependent
    "are": "hour",     # Context-dependent in time context
    "weight": "wait",  # Context-dependent
    "week": "weak",    # Context-dependent
    "wood": "would",   # Context-dependent
    "wear": "where",   # Context-dependent
    "threw": "through",
    "through": "threw", # Context-dependent
    "then": "than",    # Context-dependent
    "than": "then",    # Context-dependent
    "accept": "except", # Context-dependent
    "affect": "effect", # Context-dependent
    "who's": "whose",  # Context-dependent
    "whose": "who's",  # Context-dependent
    
    # App/Tech related
    "apk": "APK",
    "cnic": "CNIC",
    "ntu": "NTU",
    "api": "API",
    "ui": "UI",
    "ux": "UX",
}


# ============================================================================
# DOMAIN-SPECIFIC CORRECTIONS
# Educational/School App Context
# ============================================================================
EDUCATIONAL_DOMAIN_CORRECTIONS: List[CorrectionRule] = [
    # CRITICAL: "epsilon" → "absent" (voice-to-text error)
    CorrectionRule(
        wrong="epsilon",
        correct="absent",
        context_keywords=["student", "class", "present", "attendance", "mark", "one", "all"],
        priority=15
    ),
    CorrectionRule(
        wrong="one epsilon",
        correct="one absent",
        context_keywords=["student", "class", "present", "attendance"],
        priority=15
    ),
    
    # CRITICAL: "resisting" → "registration" 
    CorrectionRule(
        wrong="resisting",
        correct="registration",
        context_keywords=["app", "complete", "student", "teacher", "account", "form", "process"],
        priority=15
    ),
    CorrectionRule(
        wrong="resisting completed",
        correct="registration completed",
        context_keywords=["app", "student", "teacher", "account"],
        priority=15
    ),
    CorrectionRule(
        wrong="resisted",
        correct="registered",
        context_keywords=["app", "student", "teacher", "account", "successfully"],
        priority=15
    ),
    
    # CRITICAL: "attenders" → "attendance"
    CorrectionRule(
        wrong="attenders",
        correct="attendance",
        context_keywords=["student", "class", "school", "teacher", "present", "absent", "mark"],
        priority=15
    ),
    CorrectionRule(
        wrong="attender",
        correct="attendance",
        context_keywords=["student", "class", "school", "teacher", "present", "absent", "mark"],
        priority=15
    ),
    
    # The famous "tenders" → "attendance" fix
    CorrectionRule(
        wrong="tenders",
        correct="attendance",
        context_keywords=["student", "class", "school", "teacher", "present", "absent", "mark", "daily", "record"],
        priority=10
    ),
    CorrectionRule(
        wrong="tender",
        correct="attendance",
        context_keywords=["student", "class", "school", "teacher", "present", "absent", "mark", "daily", "record"],
        priority=10
    ),
    CorrectionRule(
        wrong="tendering",
        correct="attendance tracking",
        context_keywords=["student", "class", "school", "teacher", "record"],
        priority=10
    ),
    CorrectionRule(
        wrong="bids",
        correct="attendance records",
        context_keywords=["student", "class", "school", "teacher"],
        priority=8
    ),
    CorrectionRule(
        wrong="bidding",
        correct="attendance marking",
        context_keywords=["student", "class", "school", "teacher"],
        priority=8
    ),
    
    # Technology terms
    CorrectionRule(
        wrong="gmail",
        correct="email",
        context_keywords=["send", "receive", "notification", "message", "account"],
        priority=5
    ),
    
    # App-specific terms
    CorrectionRule(
        wrong="release apk",
        correct="release APK",
        context_keywords=["android", "app", "build", "install"],
        priority=3
    ),
    CorrectionRule(
        wrong="cnic",
        correct="CNIC",
        context_keywords=["teacher", "id", "number", "identity", "card"],
        priority=3
    ),
]


# ============================================================================
# FILLER PHRASES TO REMOVE
# ============================================================================
FILLER_PHRASES: List[str] = [
    "so i actually",
    "i actually",
    "actually i",
    "so actually",
    "like i said",
    "as i said",
    "you know",
    "i mean",
    "basically",
    "literally",
    "honestly",
    "to be honest",
    "the thing is",
    "here's the thing",
    "let me tell you",
    "so yeah",
    "yeah so",
    "okay so",
    "alright so",
    "so basically",
    "what i mean is",
    "what i'm saying is",
    "in terms of",
    "kind of",
    "sort of",
    "at the end of the day",
    "moving forward",
    "going forward",
    "in this day and age",
    "needless to say",
    "as a matter of fact",
    "at this point in time",
    "for all intents and purposes",
]


# ============================================================================
# INFORMAL TO PROFESSIONAL MAPPINGS
# ============================================================================
INFORMAL_TO_PROFESSIONAL: Dict[str, str] = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "have to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "is not",
    "y'all": "you all",
    "imma": "I am going to",
    "tryna": "trying to",
    "finna": "about to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "whatcha": "what are you",
    "gotcha": "got you",
    "outta": "out of",
    "lotta": "a lot of",
    "cuz": "because",
    "cos": "because",
    "'cause": "because",
    "tho": "though",
    "thru": "through",
    "u": "you",
    "ur": "your",
    "r": "are",
    "n": "and",
    "b4": "before",
    "2": "to",  # Context-dependent
    "4": "for", # Context-dependent
    "pls": "please",
    "plz": "please",
    "ok": "okay",
    "k": "okay",
    "ya": "yes",
    "yep": "yes",
    "nope": "no",
    "pic": "picture",
    "pics": "pictures",
    "info": "information",
    "app": "application",
    "apps": "applications",
    "msg": "message",
    "msgs": "messages",
}


class ContextCorrector:
    """
    Context-aware text correction for transcription errors.
    
    This class handles:
    1. Phonetic transcription errors (hair → her)
    2. Domain-specific corrections (tenders → attendance)
    3. Filler phrase removal
    4. Informal to professional conversion
    """
    
    def __init__(self, domain: str = "educational"):
        """
        Initialize the context corrector.
        
        Args:
            domain: The domain context for corrections
                - "educational": School/student management apps
                - "business": Business/corporate context
                - "general": General purpose corrections
        """
        self.domain = domain
        self.domain_corrections = self._get_domain_corrections(domain)
        
    def _get_domain_corrections(self, domain: str) -> List[CorrectionRule]:
        """Get domain-specific correction rules."""
        if domain == "educational":
            return EDUCATIONAL_DOMAIN_CORRECTIONS
        # Add more domains as needed
        return []
    
    def correct_transcription_errors(self, text: str) -> str:
        """
        Fix common speech-to-text transcription errors.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Text with transcription errors fixed
        """
        result = text.lower()
        
        # Apply phrase-level corrections first (more specific)
        phrase_corrections = {
            "send hair": "send her",
            "tell hair": "tell her",
            "to hair": "to her",
            "with hair": "with her",
            "for hair": "for her",
            "from hair": "from her",
            "contact hair": "contact her",
            "message hair": "message her",
            "notify hair": "notify her",
            "email hair": "email her",
            "inform hair": "inform her",
        }
        
        for wrong, correct in phrase_corrections.items():
            result = re.sub(
                r'\b' + re.escape(wrong) + r'\b',
                correct,
                result,
                flags=re.IGNORECASE
            )
        
        return result
    
    def apply_domain_corrections(self, text: str) -> str:
        """
        Apply domain-specific terminology corrections.
        
        Uses context keywords to determine if a correction should be applied.
        
        Args:
            text: Input text
            
        Returns:
            Text with domain-specific corrections
        """
        result = text
        text_lower = text.lower()
        
        # Sort by priority (highest first)
        sorted_rules = sorted(
            self.domain_corrections, 
            key=lambda x: x.priority, 
            reverse=True
        )
        
        for rule in sorted_rules:
            # Check if any context keyword is present in the text
            context_found = any(
                kw.lower() in text_lower 
                for kw in rule.context_keywords
            )
            
            if context_found:
                # Apply the correction
                pattern = r'\b' + re.escape(rule.wrong) + r'\b'
                result = re.sub(pattern, rule.correct, result, flags=re.IGNORECASE)
        
        return result
    
    def remove_filler_phrases(self, text: str) -> str:
        """
        Remove filler phrases that don't add meaning.
        
        Args:
            text: Input text
            
        Returns:
            Text with filler phrases removed
        """
        result = text
        
        # Sort by length (longest first) to handle overlapping phrases
        sorted_fillers = sorted(FILLER_PHRASES, key=len, reverse=True)
        
        for filler in sorted_fillers:
            # Match filler at start of sentence or after punctuation
            patterns = [
                r'^' + re.escape(filler) + r'\s*,?\s*',  # Start of text
                r'(?<=[.!?])\s*' + re.escape(filler) + r'\s*,?\s*',  # After sentence
                r',\s*' + re.escape(filler) + r'\s*,',  # Between commas
            ]
            
            for pattern in patterns:
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def convert_informal_to_professional(self, text: str) -> str:
        """
        Convert informal/slang words to professional alternatives.
        
        Args:
            text: Input text
            
        Returns:
            More professional text
        """
        result = text
        
        for informal, professional in INFORMAL_TO_PROFESSIONAL.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(informal) + r'\b'
            result = re.sub(pattern, professional, result, flags=re.IGNORECASE)
        
        return result
    
    def apply_all_corrections(self, text: str) -> str:
        """
        Apply all corrections in the optimal order.
        
        Order:
        1. Transcription errors (most specific)
        2. Domain-specific corrections
        3. Informal to professional
        4. Filler phrase removal
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Fully corrected text
        """
        if not text or not text.strip():
            return ""
        
        result = text
        
        # Step 1: Fix transcription errors
        result = self.correct_transcription_errors(result)
        
        # Step 2: Apply domain-specific corrections
        result = self.apply_domain_corrections(result)
        
        # Step 3: Convert informal to professional
        result = self.convert_informal_to_professional(result)
        
        # Step 4: Remove filler phrases
        result = self.remove_filler_phrases(result)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result


def add_custom_correction(wrong: str, correct: str, context_keywords: List[str], priority: int = 5):
    """
    Add a custom correction rule to the educational domain.
    
    Args:
        wrong: The incorrect word/phrase
        correct: The correct replacement
        context_keywords: Words that indicate this correction should apply
        priority: Importance (higher = more important)
    """
    EDUCATIONAL_DOMAIN_CORRECTIONS.append(
        CorrectionRule(
            wrong=wrong,
            correct=correct,
            context_keywords=context_keywords,
            priority=priority
        )
    )


# ============================================================================
# TEXT SECTIONING - Divide text into readable parts
# ============================================================================

# Keywords that indicate a new section should start
SECTION_KEYWORDS = [
    # Step indicators
    "first", "second", "third", "fourth", "fifth",
    "firstly", "secondly", "thirdly",
    "step one", "step two", "step three", "step 1", "step 2", "step 3",
    "next", "then", "after that", "finally", "lastly",
    
    # Topic transitions
    "now", "moving on", "let's talk about", "regarding",
    "as for", "when it comes to", "in terms of",
    "another thing", "another feature", "one more thing",
    
    # Process indicators
    "to start", "to begin", "starting with",
    "once you", "when you", "after you",
    "if you", "you can", "you need to", "you have to",
    
    # Feature/section indicators
    "the first feature", "the second feature",
    "the main", "an important", "a key",
    "for example", "for instance",
    
    # App-specific
    "login", "registration", "password", "account",
    "dashboard", "home screen", "main screen",
    "settings", "profile", "notification",
]


def divide_into_sections(text: str, min_section_length: int = 100, max_section_length: int = 500) -> List[str]:
    """
    Divide a long text into readable sections based on topic transitions.
    
    Args:
        text: The input text to divide
        min_section_length: Minimum characters per section
        max_section_length: Maximum characters per section
        
    Returns:
        List of text sections
    """
    if not text or len(text) < min_section_length:
        return [text] if text else []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    sections = []
    current_section = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if this sentence starts a new section
        starts_new_section = False
        sentence_lower = sentence.lower()
        
        # Check for section keywords
        for keyword in SECTION_KEYWORDS:
            if sentence_lower.startswith(keyword) or f" {keyword}" in sentence_lower[:50]:
                starts_new_section = True
                break
        
        # Force new section if current is too long
        if current_length > max_section_length:
            starts_new_section = True
        
        # Start new section if needed (but not if current is too short)
        if starts_new_section and current_length >= min_section_length:
            sections.append(' '.join(current_section))
            current_section = []
            current_length = 0
        
        current_section.append(sentence)
        current_length += len(sentence)
    
    # Add remaining section
    if current_section:
        sections.append(' '.join(current_section))
    
    return sections


def format_as_structured_text(text: str, add_headers: bool = True) -> str:
    """
    Convert raw text into a structured, readable format with sections.
    
    Args:
        text: The input text
        add_headers: Whether to add section headers
        
    Returns:
        Structured text with sections and headers
    """
    sections = divide_into_sections(text)
    
    if len(sections) <= 1:
        return text
    
    formatted_parts = []
    
    # Section header templates
    headers = [
        "Introduction",
        "Getting Started",
        "Main Features",
        "Step-by-Step Guide",
        "Key Points",
        "Additional Features",
        "Important Notes",
        "Summary",
        "Conclusion",
    ]
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
            
        if add_headers and i < len(headers):
            # Try to detect what the section is about
            section_header = detect_section_topic(section) or headers[min(i, len(headers)-1)]
            formatted_parts.append(f"\n## {section_header}\n\n{section}")
        else:
            formatted_parts.append(f"\n{section}")
    
    return '\n'.join(formatted_parts).strip()


def detect_section_topic(section: str) -> Optional[str]:
    """
    Try to detect what topic a section covers.
    
    Args:
        section: The section text
        
    Returns:
        A topic header or None
    """
    section_lower = section.lower()
    
    topic_keywords = {
        "Introduction": ["welcome", "today", "this video", "this tutorial", "i'm going to", "we will"],
        "Login Process": ["login", "sign in", "log in", "username", "password"],
        "Registration": ["registration", "sign up", "create account", "register", "new account"],
        "Dashboard Overview": ["dashboard", "home screen", "main screen", "after login"],
        "Attendance System": ["attendance", "present", "absent", "mark", "daily"],
        "User Profile": ["profile", "account settings", "your information"],
        "Settings": ["settings", "preferences", "configure", "options"],
        "Features": ["feature", "functionality", "you can", "allows you"],
        "Important Notes": ["important", "note that", "remember", "make sure", "don't forget"],
        "Conclusion": ["finally", "in conclusion", "to summarize", "that's all", "thank you"],
    }
    
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in section_lower[:200]:  # Check first 200 chars
                return topic
    
    return None


def clean_and_section_text(text: str, domain: str = "educational") -> str:
    """
    Apply corrections AND divide into sections for maximum readability.
    
    This is the main function to use for complete text processing.
    
    Args:
        text: Raw transcribed text
        domain: The domain context
        
    Returns:
        Clean, corrected, and sectioned text
    """
    # First, apply all corrections
    corrector = ContextCorrector(domain=domain)
    corrected = corrector.apply_all_corrections(text)
    
    # Then, divide into sections
    sectioned = format_as_structured_text(corrected, add_headers=True)
    
    return sectioned


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    corrector = ContextCorrector(domain="educational")
    
    test_cases = [
        "I will send hair the message about student tenders",
        "The app shows daily tenders for all students in the class",
        "So I actually write this format for the teacher to mark tenders",
        "Gmail me the student attendance report",
        "gonna add the feature to track student tendering",
    ]
    
    print("Context Correction Tests")
    print("=" * 60)
    
    for test in test_cases:
        corrected = corrector.apply_all_corrections(test)
        print(f"\nOriginal: {test}")
        print(f"Corrected: {corrected}")
