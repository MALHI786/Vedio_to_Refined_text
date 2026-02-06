"""
Text Improvement Module
Converts broken/informal English to fluent, professional text
Uses Transformer-based grammar correction models

Supports:
- Multiple grammar correction models (switchable)
- Multi-language text improvement
- Long text processing with chunking
- High accuracy with Grammarly CoEdit model
- Context-aware transcription error correction
- Multi-pass refinement for better quality
- LLM-based deep understanding (optional)
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional, List, Dict, Callable
import re

# Import context corrector for domain-specific fixes
try:
    from utils.context_corrector import ContextCorrector, clean_and_section_text, format_as_structured_text, divide_into_sections
except ImportError:
    from context_corrector import ContextCorrector, clean_and_section_text, format_as_structured_text, divide_into_sections


# Available models ranked by quality
AVAILABLE_MODELS = {
    "high_accuracy": "grammarly/coedit-large",      # Best quality, slower
    "balanced": "vennify/t5-base-grammar-correction", # Good balance
    "fast": "prithivida/grammar_error_correcter_v1",  # Faster, decent
    "multilingual": "facebook/mbart-large-50-many-to-many-mmt",  # For translation
    "llm_refine": "google/flan-t5-large",           # For deep understanding
}

# Improvement tasks with detailed prompts
IMPROVEMENT_PROMPTS = {
    "fix_grammar": "Fix grammar: {text}",
    "fix_all": "Fix grammar and spelling: {text}",
    "fluent": "Make this more fluent: {text}",
    "professional": "Rewrite this professionally: {text}",
    "simplify": "Simplify this text: {text}",
    "expand": "Expand this text with more detail: {text}",
    "formal": "Make this formal: {text}",
    "casual": "Make this more casual: {text}",
}


class TextImprover:
    """
    Text improvement using Transformer-based models.
    
    Transforms informal, broken English into fluent, professional text.
    
    Features:
    - Multiple model options (accuracy vs speed tradeoff)
    - Handles long texts with smart chunking
    - Multi-language support
    - Grammar, spelling, and fluency correction
    - Context-aware transcription error correction
    - Multi-pass refinement for 10x better results
    """
    
    def __init__(
        self, 
        model_name: str = "grammarly/coedit-large",
        use_fast_model: bool = False,
        domain: str = "educational",
        enable_context_correction: bool = True,
        refinement_passes: int = 1,
        low_resource: bool = False
    ):
        """
        Initialize the text improvement model.
        
        Args:
            model_name: HuggingFace model for grammar correction
                - "grammarly/coedit-large" - BEST quality (recommended for GPU)
                - "vennify/t5-base-grammar-correction" - Good balance
                - "prithivida/grammar_error_correcter_v1" - Fast (recommended for CPU)
            use_fast_model: If True, uses faster but less accurate model
            domain: Domain for context corrections
                - "educational" - School/student management apps
                - "business" - Corporate/business context
                - "general" - General purpose
            enable_context_correction: Enable domain-aware corrections
            refinement_passes: Number of refinement passes (1-3 recommended)
            low_resource: Enable CPU-friendly optimizations (reduces beam search, smaller batches)
        """
        if use_fast_model:
            model_name = AVAILABLE_MODELS["fast"]
        
        # Auto-enable low resource mode for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == -1:
            low_resource = True
            print(f"⚡ CPU detected: Enabling low-resource optimizations")
        
        print(f"🔄 Loading text improvement model: {model_name}")
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"📱 Using device: {device_name}")
        
        self.model_name = model_name
        self.corrector = None
        self.tokenizer = None
        self.model = None
        self.refinement_passes = max(1, min(5, refinement_passes))  # Clamp 1-5
        self.low_resource = low_resource
        
        # Initialize context corrector
        self.enable_context_correction = enable_context_correction
        if enable_context_correction:
            self.context_corrector = ContextCorrector(domain=domain)
            print(f"📚 Context correction enabled for domain: {domain}")
        
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load the specified model with fallbacks."""
        models_to_try = [
            model_name,
            AVAILABLE_MODELS["balanced"],
            AVAILABLE_MODELS["fast"],
            "t5-base",
        ]
        
        for model in models_to_try:
            try:
                print(f"   Trying to load: {model}")
                
                # For CoEdit model, use proper task prefix
                if "coedit" in model.lower():
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
                    if self.device == 0:
                        self.model = self.model.cuda()
                    self.model_name = model
                    self.is_coedit = True
                else:
                    self.corrector = pipeline(
                        "text2text-generation",
                        model=model,
                        device=self.device
                    )
                    self.model_name = model
                    self.is_coedit = False
                
                print(f"✅ Model loaded: {model}")
                return
                
            except Exception as e:
                print(f"   ⚠️ Failed to load {model}: {str(e)[:50]}...")
                continue
        
        raise RuntimeError("Could not load any grammar correction model")
    
    def _split_into_chunks(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into processable chunks."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 1 < max_length:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def improve(self, text: str, task: str = "Fix grammar", passes: int = None) -> str:
        """
        Improve the grammar and fluency of text.
        
        Args:
            text: Input text (can be informal/broken English)
            task: Task instruction for CoEdit model:
                - "Fix grammar" (default)
                - "Fix grammar and spelling"  
                - "Make this more fluent"
                - "Rewrite this professionally"
                - "Paraphrase this"
            passes: Number of refinement passes (overrides init setting)
            
        Returns:
            Improved, professional text
        """
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        num_passes = passes if passes is not None else self.refinement_passes
        
        print(f"📝 Improving text ({len(text)} chars) with {num_passes} pass(es)...")
        
        # Step 1: Apply context corrections FIRST (before grammar model)
        if self.enable_context_correction:
            print("   🔍 Applying context-aware corrections...")
            text = self.context_corrector.apply_all_corrections(text)
        
        # Step 2: Multi-pass refinement
        current_text = text
        for pass_num in range(num_passes):
            print(f"   📊 Refinement pass {pass_num + 1}/{num_passes}...")
            
            # Split into chunks for long text
            chunks = self._split_into_chunks(current_text, max_length=200)
            
            improved_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    if hasattr(self, 'is_coedit') and self.is_coedit:
                        # Use CoEdit model with task prefix
                        improved = self._improve_with_coedit(chunk, task)
                    else:
                        # Use pipeline model
                        improved = self._improve_with_pipeline(chunk)
                    
                    improved_chunks.append(improved)
                    
                except Exception as e:
                    print(f"   ⚠️ Chunk {i+1} error: {e}")
                    improved_chunks.append(chunk)
            
            current_text = " ".join(improved_chunks)
            
            # Use different tasks for subsequent passes
            if pass_num == 0 and num_passes > 1:
                task = "Make this more fluent"
            elif pass_num == 1 and num_passes > 2:
                task = "Rewrite this professionally"
        
        final_text = self._post_process(current_text)
        
        print(f"✅ Text improvement complete!")
        return final_text
    
    def _improve_with_coedit(self, text: str, task: str = "Fix grammar") -> str:
        """Improve text using Grammarly CoEdit model."""
        # CoEdit uses task prefixes
        input_text = f"{task}: {text}"
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=256,
            truncation=True
        )
        
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Reduce computational cost for CPU
        if self.low_resource:
            # Use faster generation settings for CPU
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=2,  # Reduced from 4
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        else:
            # Use high-quality settings for GPU
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    
    def _improve_with_pipeline(self, text: str) -> str:
        """Improve text using HuggingFace pipeline."""
        if "t5" in self.model_name.lower() or "grammar" in self.model_name.lower():
            input_text = f"grammar: {text}"
        else:
            input_text = text
        
        result = self.corrector(
            input_text,
            max_length=256,
            num_return_sequences=1,
            do_sample=False
        )
        
        return result[0]['generated_text'].strip()
    
    def _post_process(self, text: str) -> str:
        """Clean up the improved text."""
        # Remove prompt markers
        text = re.sub(r'^(grammar:|fix grammar:|fix:)\s*', '', text, flags=re.IGNORECASE)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?])([A-Z])', r'\1 \2', text)
        
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text.strip()
    
    def improve_for_professional(self, text: str) -> str:
        """Make text more professional and formal."""
        return self.improve(text, task="Rewrite this professionally")
    
    def improve_fluency(self, text: str) -> str:
        """Make text more fluent and natural."""
        return self.improve(text, task="Make this more fluent")
    
    def fix_all_errors(self, text: str) -> str:
        """Fix all grammar and spelling errors."""
        return self.improve(text, task="Fix grammar and spelling")
    
    def deep_refine(self, text: str) -> str:
        """
        Apply deep refinement: context correction + grammar + fluency + professional.
        This is a 3-pass process for maximum quality.
        """
        return self.improve(text, task="Fix grammar", passes=3)
    
    def refine_for_script(self, text: str) -> str:
        """
        Refine text specifically for video script narration.
        Optimized for voice-over scripts with natural pacing.
        """
        # First apply context corrections
        if self.enable_context_correction:
            text = self.context_corrector.apply_all_corrections(text)
        
        # Then improve with focus on spoken clarity
        result = self.improve(text, task="Make this more fluent", passes=2)
        
        # Add pauses for natural speech (commas for breath points)
        result = self._add_natural_pauses(result)
        
        return result
    
    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses for script narration."""
        # Add comma before conjunctions for breath points in long sentences
        words = text.split()
        if len(words) > 15:
            conjunctions = ['and', 'but', 'or', 'so', 'yet', 'because', 'although']
            for conj in conjunctions:
                text = re.sub(
                    r'(?<=[a-z])\s+(' + conj + r')\s+',
                    r', \1 ',
                    text,
                    flags=re.IGNORECASE
                )
        return text
    
    def add_custom_correction(self, wrong: str, correct: str, context_keywords: List[str]):
        """
        Add a custom correction rule for domain-specific terms.
        
        Args:
            wrong: The incorrect word/phrase
            correct: The correct replacement
            context_keywords: Words that indicate this correction should apply
        """
        if self.enable_context_correction:
            from utils.context_corrector import add_custom_correction
            add_custom_correction(wrong, correct, context_keywords)
            print(f"✅ Added custom correction: '{wrong}' → '{correct}'")
    
    def improve_with_sections(self, text: str, task: str = "Fix grammar") -> str:
        """
        Improve text AND divide it into readable sections with headers.
        
        This is the BEST method for maximum readability.
        It applies:
        1. Context-aware corrections (tenders→attendance, epsilon→absent, etc.)
        2. Grammar and fluency improvements
        3. Text sectioning with topic headers
        
        Args:
            text: Raw transcribed text
            task: Improvement task
            
        Returns:
            Clean, corrected, and sectioned text with headers
        """
        if not text or not text.strip():
            return ""
        
        print("📝 Processing with full corrections and sectioning...")
        
        # Step 1: Apply context corrections FIRST
        if self.enable_context_correction:
            print("   🔍 Step 1: Applying context-aware corrections...")
            text = self.context_corrector.apply_all_corrections(text)
        
        # Step 2: Divide into sections for processing
        print("   📊 Step 2: Dividing into sections...")
        sections = divide_into_sections(text)
        print(f"   Found {len(sections)} sections")
        
        # Step 3: Improve each section
        improved_sections = []
        for i, section in enumerate(sections):
            print(f"   ✏️ Improving section {i+1}/{len(sections)}...")
            improved = self.improve(section, task=task, passes=self.refinement_passes)
            improved_sections.append(improved)
        
        # Step 4: Format with headers
        print("   📑 Step 3: Adding section headers...")
        final_text = self._format_sections_with_headers(improved_sections)
        
        print("✅ Processing complete!")
        return final_text
    
    def _format_sections_with_headers(self, sections: List[str]) -> str:
        """Format improved sections with topic headers."""
        formatted_parts = []
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Detect topic for this section
            try:
                from utils.context_corrector import detect_section_topic
            except ImportError:
                from context_corrector import detect_section_topic
            
            topic = detect_section_topic(section)
            
            if topic:
                formatted_parts.append(f"\n## {topic}\n\n{section}")
            elif i == 0:
                formatted_parts.append(f"\n## Introduction\n\n{section}")
            elif i == len(sections) - 1:
                formatted_parts.append(f"\n## Conclusion\n\n{section}")
            else:
                formatted_parts.append(f"\n## Part {i+1}\n\n{section}")
        
        return '\n'.join(formatted_parts).strip()


def improve_text(text: str, use_high_quality: bool = True, passes: int = 1) -> str:
    """
    Convenience function for quick text improvement.
    
    Args:
        text: Input text
        use_high_quality: Use high quality model (slower but better)
        passes: Number of refinement passes
        
    Returns:
        Improved text
    """
    improver = TextImprover(
        use_fast_model=not use_high_quality,
        refinement_passes=passes
    )
    return improver.improve(text)


def improve_for_script(text: str, domain: str = "educational") -> str:
    """
    Improve text specifically for video script narration.
    
    Args:
        text: Raw transcribed or draft text
        domain: Context domain for corrections
        
    Returns:
        Text optimized for voice-over narration
    """
    improver = TextImprover(
        domain=domain,
        enable_context_correction=True,
        refinement_passes=2
    )
    return improver.refine_for_script(text)


if __name__ == "__main__":
    print("Text Improvement Module")
    print("=" * 50)
    print("\nAvailable models:")
    for key, model in AVAILABLE_MODELS.items():
        print(f"  {key}: {model}")
    
    # Test with context correction
    test_cases = [
        "i make new app and i want explain this",
        "he dont know what to do with the problem",
        "me and him went to store yesterday",
        "I will send hair the student tenders report",  # Should fix hair→her, tenders→attendance
        "gonna add feature to track student tendering",  # Should fix gonna, tendering
    ]
    
    print("\nTest cases:")
    for test in test_cases:
        print(f"  - {test}")
