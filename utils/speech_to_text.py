"""
Speech-to-Text Module
Converts audio to text using OpenAI Whisper (Neural Network)
Supports: Multiple languages, long videos (10+ minutes), auto language detection
"""

import os
import whisper
import torch
from typing import Optional, Dict, Any, List


def setup_ffmpeg_path():
    """
    Add FFmpeg to system PATH if not already available.
    This is required because Whisper uses FFmpeg internally to load audio.
    """
    # Check if ffmpeg is already in PATH
    import shutil
    if shutil.which("ffmpeg"):
        return
    
    # Common FFmpeg installation locations on Windows
    ffmpeg_locations = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.expanduser("~\\ffmpeg\\bin"),
    ]
    
    for location in ffmpeg_locations:
        if os.path.exists(location):
            ffmpeg_exe = os.path.join(location, "ffmpeg.exe")
            if os.path.exists(ffmpeg_exe):
                # Add to PATH
                os.environ["PATH"] = location + os.pathsep + os.environ.get("PATH", "")
                print(f"📍 Added FFmpeg to PATH: {location}")
                return
    
    # Check WinGet packages directory for ffmpeg
    winget_packages = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_packages):
        for folder in os.listdir(winget_packages):
            if "ffmpeg" in folder.lower():
                ffmpeg_dir = os.path.join(winget_packages, folder)
                # Search for ffmpeg.exe recursively
                for root, dirs, files in os.walk(ffmpeg_dir):
                    if "ffmpeg.exe" in files:
                        os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                        print(f"📍 Added FFmpeg to PATH: {root}")
                        return


# Setup FFmpeg on module import
setup_ffmpeg_path()


# Supported languages with their codes
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'ur': 'Urdu',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'tr': 'Turkish',
    'it': 'Italian',
}


class SpeechToText:
    """
    Speech-to-Text converter using OpenAI Whisper.
    
    Whisper is a transformer-based neural network trained on 
    680,000 hours of multilingual and multitask supervised data.
    
    Features:
    - Multi-language support (99+ languages)
    - Auto language detection
    - Long video support (chunked processing)
    - Timestamp generation
    """
    
    # Available models: tiny, base, small, medium, large
    AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the Whisper model.
        
        Args:
            model_name: Size of Whisper model to use
                - tiny: Fastest, least accurate (39M params)
                - base: Good balance (74M params) - recommended
                - small: Better accuracy (244M params)
                - medium: High accuracy (769M params)
                - large: Best accuracy (1550M params) - requires GPU
        """
        if model_name not in self.AVAILABLE_MODELS:
            print(f"⚠️ Model '{model_name}' not in standard list, attempting to load...")
        
        print(f"🔄 Loading Whisper '{model_name}' model...")
        
        # Detect device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Using device: {self.device.upper()}")
        
        # Load the model
        self.model = whisper.load_model(model_name, device=self.device)
        self.model_name = model_name
        
        print(f"✅ Whisper model loaded successfully!")
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        task: str = "transcribe",
        detect_language: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text with multi-language support.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'ur', 'hi') or None for auto-detect
            task: 'transcribe' (keep original language) or 'translate' (to English)
            detect_language: Auto-detect language if not specified
            
        Returns:
            Dictionary containing:
                - text: Full transcribed text
                - segments: List of segments with timestamps
                - language: Detected/used language code
                - language_name: Human-readable language name
        """
        print(f"🎤 Transcribing audio: {audio_path}")
        
        # Build transcription options
        options = {
            "task": task,
            "fp16": (self.device == "cuda"),
            "verbose": False,
        }
        
        # Handle language setting
        if language:
            options["language"] = language
            print(f"📝 Using specified language: {SUPPORTED_LANGUAGES.get(language, language)}")
        elif detect_language:
            print(f"🔍 Auto-detecting language...")
        
        # Transcribe
        result = self.model.transcribe(audio_path, **options)
        
        detected_lang = result.get("language", "en")
        lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang.upper())
        
        print(f"✅ Transcription complete! Language: {lang_name}")
        
        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": detected_lang,
            "language_name": lang_name
        }
    
    def transcribe_multilingual(
        self,
        audio_path: str,
        translate_to_english: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio with automatic language detection.
        Optimized for mixed-language content.
        
        Args:
            audio_path: Path to audio file
            translate_to_english: If True, translates to English
            
        Returns:
            Transcription result with detected language
        """
        task = "translate" if translate_to_english else "transcribe"
        return self.transcribe(
            audio_path,
            language=None,  # Auto-detect
            task=task,
            detect_language=True
        )
    
    def get_text(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Simple method to get just the transcribed text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Transcribed text string
        """
        result = self.transcribe(audio_path, language=language)
        return result["text"]
    
    def get_segments_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Get transcription with timestamps for each segment.
        Useful for subtitles or long video analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments with start, end, and text
        """
        result = self.transcribe(audio_path)
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]


def speech_to_text(
    audio_path: str, 
    model_name: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Convenience function for quick transcription.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use
        language: Optional language code (None for auto-detect)
        
    Returns:
        Transcribed text
    """
    stt = SpeechToText(model_name)
    return stt.get_text(audio_path, language=language)


if __name__ == "__main__":
    print("Speech-to-Text Module (Whisper)")
    print("=" * 50)
    print("\nSupported Languages:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"  {code}: {name}")
    print("\nAvailable models:", SpeechToText.AVAILABLE_MODELS)
    print("Device available:", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
