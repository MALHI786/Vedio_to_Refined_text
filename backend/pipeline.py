"""
AI Pipeline Module
Complete pipeline: Video → Audio → Text → Improved Text

Features:
- Multi-language support (auto-detection)
- Long video support (10+ minutes)
- High-quality grammar correction
- Chunked processing for memory efficiency
- Audio splitting for large files
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from utils.audio_extractor import extract_audio, get_audio_info
from utils.speech_to_text import SpeechToText, SUPPORTED_LANGUAGES
from utils.text_cleaner import clean_text
from utils.text_improver import TextImprover

# Import audio chunker for large files
try:
    from utils.audio_chunker import AudioChunker, split_audio_into_chunks, combine_transcriptions, cleanup_chunks
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False
    print("⚠️ Audio chunker not available, large files may fail")


class AIPipeline:
    """
    Complete AI Pipeline for Video-to-Fluent-Text conversion.
    
    Pipeline stages:
    1. Video → Audio (FFmpeg)
    2. Audio → Raw Text (Whisper Neural Network) - with chunking for large files
    3. Raw Text → Cleaned Text (Text Processing)
    4. Cleaned Text → Fluent Text (Transformer Model)
    
    Features:
    - Supports 99+ languages with auto-detection
    - Handles videos up to 10+ minutes
    - High-accuracy grammar correction with Grammarly CoEdit
    """
    
    def __init__(
        self, 
        whisper_model: str = "base",
        grammar_model: str = "grammarly/coedit-large",
        use_fast_grammar: bool = False,
        domain: str = "educational",
        refinement_passes: int = 2,
        low_resource: bool = False
    ):
        """
        Initialize the AI Pipeline.
        
        Args:
            whisper_model: Whisper model size
                - 'tiny': Fastest, least accurate
                - 'base': Good balance (recommended)
                - 'small': Better accuracy
                - 'medium': High accuracy
                - 'large': Best accuracy (needs GPU)
            grammar_model: HuggingFace model for grammar correction
            use_fast_grammar: Use faster but less accurate grammar model
            domain: Domain for context-aware corrections
                - 'educational': School/student management apps
                - 'business': Corporate/business context
                - 'general': General purpose
            refinement_passes: Number of refinement passes (1-3 recommended)
            low_resource: Enable CPU-friendly optimizations
        """
        print("=" * 60)
        print("🚀 Initializing AI Pipeline")
        print("=" * 60)
        
        # Initialize Speech-to-Text model
        self.stt = SpeechToText(model_name=whisper_model)
        
        # Initialize Text Improver model with context correction
        self.improver = TextImprover(
            model_name=grammar_model,
            use_fast_model=use_fast_grammar,
            domain=domain,
            enable_context_correction=True,
            refinement_passes=refinement_passes,
            low_resource=low_resource
        )
        
        print("=" * 60)
        print("✅ AI Pipeline ready!")
        print("=" * 60)
    
    def _transcribe_with_chunking(
        self,
        audio_path: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
        chunk_duration: int = 180  # 3 minutes
    ) -> Dict[str, Any]:
        """
        Transcribe audio with chunking support for large files.
        
        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            translate_to_english: Translate to English
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Transcription result with combined text
        """
        # Check if chunking is available and needed
        if not CHUNKING_AVAILABLE:
            print("   ⚠️ Chunking not available, using direct transcription")
            task = "translate" if translate_to_english else "transcribe"
            return self.stt.transcribe(audio_path, language=language, task=task)
        
        # Get audio duration
        from utils.audio_chunker import get_audio_duration
        duration = get_audio_duration(audio_path)
        
        # If short enough, use direct transcription
        if duration < chunk_duration + 30:
            print(f"   Audio is {duration:.0f}s, using direct transcription")
            task = "translate" if translate_to_english else "transcribe"
            return self.stt.transcribe(audio_path, language=language, task=task)
        
        # Use chunked processing
        print(f"   📊 Audio is {duration:.0f}s ({duration/60:.1f} min), using chunked processing")
        
        # Split audio into chunks
        output_dir = str(Path(audio_path).parent)
        chunk_paths = split_audio_into_chunks(
            audio_path,
            output_dir=output_dir,
            chunk_duration=chunk_duration,
            overlap=5
        )
        
        # Transcribe each chunk
        transcriptions = []
        detected_language = None
        
        task = "translate" if translate_to_english else "transcribe"
        
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n   🎤 Transcribing chunk {i+1}/{len(chunk_paths)}...")
            try:
                result = self.stt.transcribe(
                    chunk_path,
                    language=language or detected_language,
                    task=task
                )
                transcriptions.append(result["text"])
                
                # Use detected language for subsequent chunks
                if detected_language is None:
                    detected_language = result.get("language", "en")
                    
            except Exception as e:
                print(f"   ❌ Chunk {i+1} error: {e}")
                transcriptions.append("")
        
        # Combine transcriptions
        print("\n   🔗 Combining chunk transcriptions...")
        combined_text = combine_transcriptions(transcriptions)
        
        # Cleanup chunks
        if chunk_paths[0] != audio_path:
            cleanup_chunks(chunk_paths)
            print("   🗑️ Cleaned up temporary chunks")
        
        return {
            "text": combined_text,
            "language": detected_language or "en",
            "language_name": SUPPORTED_LANGUAGES.get(detected_language, "English"),
            "segments": []
        }
    
    def process_video(
        self, 
        video_path: str,
        output_dir: str = None,
        keep_audio: bool = False,
        language: Optional[str] = None,
        translate_to_english: bool = False,
        task: str = "Fix grammar"
    ) -> Dict[str, Any]:
        """
        Process a video file through the complete pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for temporary files
            keep_audio: Whether to keep extracted audio file
            language: Language code (None for auto-detect)
                - 'en': English
                - 'ur': Urdu
                - 'hi': Hindi
                - 'ar': Arabic
                - etc.
            translate_to_english: If True, translates to English
            task: Type of text improvement
                - 'Fix grammar': Basic grammar correction
                - 'Make this more fluent': Fluency improvement
                - 'Rewrite this professionally': Professional tone
            
        Returns:
            Dictionary containing:
                - video_path: Original video path
                - audio_path: Extracted audio path (if kept)
                - raw_text: Original transcribed text
                - cleaned_text: Text after cleaning
                - improved_text: Final fluent text
                - detected_language: Language detected/used
                - audio_info: Audio metadata
        """
        print("\n" + "=" * 60)
        print("🎬 Processing Video")
        print("=" * 60)
        
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = str(Path(video_path).parent)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate audio path
        video_name = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
        
        result = {
            "video_path": video_path,
            "audio_path": None,
            "raw_text": "",
            "cleaned_text": "",
            "improved_text": "",
            "script_text": "",
            "detected_language": "",
            "language_name": "",
            "audio_info": {}
        }
        
        try:
            # Stage 1: Extract Audio
            print("\n📍 Stage 1: Extracting Audio...")
            audio_path = extract_audio(video_path, audio_path)
            result["audio_info"] = get_audio_info(audio_path)
            
            duration = result["audio_info"].get("duration", 0)
            if duration > 0:
                print(f"   Duration: {result['audio_info'].get('duration_formatted', f'{duration:.1f}s')}")
            
            # Stage 2: Speech to Text (with chunking for long audio)
            print("\n📍 Stage 2: Converting Speech to Text...")
            
            # Use chunked transcription for large files
            transcription = self._transcribe_with_chunking(
                audio_path,
                language=language,
                translate_to_english=translate_to_english,
                chunk_duration=180  # 3 minute chunks
            )
            
            result["raw_text"] = transcription["text"]
            result["detected_language"] = transcription.get("language", "en")
            result["language_name"] = transcription.get("language_name", "English")
            
            print(f"   Language: {result['language_name']}")
            print(f"   Raw text: {result['raw_text'][:100]}...")
            
            # Stage 3: Clean Text
            print("\n📍 Stage 3: Cleaning Text...")
            result["cleaned_text"] = clean_text(result["raw_text"])
            print(f"   Cleaned text: {result['cleaned_text'][:100]}...")
            
            # Stage 4: Improve Text (only for English text)
            print("\n📍 Stage 4: Improving Text...")
            if result["detected_language"] == "en" or translate_to_english:
                # Use sectioned improvement for maximum readability
                result["improved_text"] = self.improver.improve_with_sections(result["cleaned_text"], task=task)
                
                # Generate script-ready version too (non-sectioned for voiceover)
                print("   🎭 Generating script-ready version...")
                result["script_text"] = self.improver.refine_for_script(result["cleaned_text"])
            else:
                # For non-English, skip grammar correction or translate first
                print(f"   ⚠️ Grammar improvement works best with English text")
                result["improved_text"] = result["cleaned_text"]
                result["script_text"] = result["cleaned_text"]
            
            print(f"   Improved text: {result['improved_text'][:100]}...")
            
            # Cleanup
            if keep_audio:
                result["audio_path"] = audio_path
            else:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print("\n🗑️ Temporary audio file removed")
            
            print("\n" + "=" * 60)
            print("✅ Processing Complete!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise
        
        return result
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Process an audio file directly (skip video extraction).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Same as process_video but without video info
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        print("\n" + "=" * 60)
        print("🎤 Processing Audio")
        print("=" * 60)
        
        result = {
            "audio_path": audio_path,
            "raw_text": "",
            "cleaned_text": "",
            "improved_text": "",
            "audio_info": get_audio_info(audio_path)
        }
        
        # Stage 1: Speech to Text
        print("\n📍 Stage 1: Converting Speech to Text...")
        transcription = self.stt.transcribe(audio_path)
        result["raw_text"] = transcription["text"]
        
        # Stage 2: Clean Text
        print("\n📍 Stage 2: Cleaning Text...")
        result["cleaned_text"] = clean_text(result["raw_text"])
        
        # Stage 3: Improve Text
        print("\n📍 Stage 3: Improving Text...")
        result["improved_text"] = self.improver.improve(result["cleaned_text"])
        
        print("\n" + "=" * 60)
        print("✅ Processing Complete!")
        print("=" * 60)
        
        return result
    
    def process_text(self, text: str, for_script: bool = False) -> Dict[str, str]:
        """
        Process raw text directly (skip audio extraction and transcription).
        
        Args:
            text: Raw text to process
            for_script: If True, optimize output for video narration
            
        Returns:
            Dictionary with cleaned and improved text
        """
        print("\n📍 Processing text...")
        
        cleaned = clean_text(text)
        
        if for_script:
            # Use script-optimized refinement
            improved = self.improver.refine_for_script(cleaned)
        else:
            improved = self.improver.improve(cleaned)
        
        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "improved_text": improved
        }
    
    def deep_refine(self, text: str) -> str:
        """
        Apply deep refinement with maximum passes for highest quality.
        Use this when quality is more important than speed.
        
        Args:
            text: Text to refine
            
        Returns:
            Highly refined text
        """
        return self.improver.deep_refine(text)
    
    def iterative_refine(self, text: str, iterations: int = 3) -> str:
        """
        Apply multiple refinement iterations for best quality.
        Stops early if no further improvements are detected.
        
        Args:
            text: Text to refine
            iterations: Number of iterations (3 recommended)
            
        Returns:
            Best quality refined text
        """
        current_text = text
        
        for i in range(iterations):
            print(f"\n🔄 Iteration {i+1}/{iterations}")
            result = self.improver.improve(current_text)
            
            # Stop if no changes
            if result == current_text:
                print("   ✅ No further improvements needed!")
                break
            
            current_text = result
            print(f"   📝 Length: {len(current_text)} chars")
        
        return current_text
    
    def add_custom_correction(self, wrong: str, correct: str, context_keywords: List[str]):
        """
        Add a custom domain-specific correction.
        
        Args:
            wrong: The incorrect word/phrase
            correct: The correct replacement
            context_keywords: Words that indicate this correction should apply
        """
        self.improver.add_custom_correction(wrong, correct, context_keywords)


def process_video(video_path: str) -> Dict[str, Any]:
    """
    Convenience function to process a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Processing results
    """
    pipeline = AIPipeline()
    return pipeline.process_video(video_path)


def refine_script(text: str, domain: str = "educational") -> str:
    """
    Convenience function to refine text for video script narration.
    
    Args:
        text: Raw or draft script text
        domain: Context domain for corrections
        
    Returns:
        Script-ready refined text
    """
    pipeline = AIPipeline(domain=domain, refinement_passes=2)
    result = pipeline.process_text(text, for_script=True)
    return result["improved_text"]


if __name__ == "__main__":
    print("AI Pipeline Module")
    print("=" * 60)
    print("\nUsage:")
    print("  pipeline = AIPipeline()")
    print("  result = pipeline.process_video('video.mp4')")
    print("\nOr for text-only:")
    print("  result = pipeline.process_text('your text here')")
