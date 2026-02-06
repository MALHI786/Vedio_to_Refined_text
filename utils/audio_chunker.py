"""
Audio Chunking Module
Splits large audio files into manageable chunks for processing
Then combines the results

Features:
- Automatic chunk splitting for long audio (>5 minutes)
- Overlap between chunks to avoid cutting words
- Memory-efficient processing
- Preserves timing information
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import math


def get_ffmpeg_path() -> str:
    """Find FFmpeg executable path."""
    possible_paths = [
        "ffmpeg",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    
    for path in possible_paths:
        if path == "ffmpeg":
            try:
                result = subprocess.run(
                    ["ffmpeg", "-version"], 
                    capture_output=True, 
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                if result.returncode == 0:
                    return "ffmpeg"
            except FileNotFoundError:
                continue
        elif os.path.exists(path):
            return path
    
    return "ffmpeg"


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    ffprobe_path = get_ffmpeg_path().replace("ffmpeg", "ffprobe")
    
    cmd = [
        ffprobe_path, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        pass
    
    return 0.0


def split_audio_into_chunks(
    audio_path: str,
    output_dir: str = None,
    chunk_duration: int = 180,  # 3 minutes per chunk
    overlap: int = 5  # 5 seconds overlap
) -> List[str]:
    """
    Split a large audio file into smaller chunks.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory for chunk files (default: same as audio)
        chunk_duration: Duration of each chunk in seconds (default: 180s = 3min)
        overlap: Overlap between chunks in seconds (to avoid cutting words)
        
    Returns:
        List of paths to chunk files
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get audio duration
    total_duration = get_audio_duration(audio_path)
    if total_duration == 0:
        print("⚠️ Could not determine audio duration, processing as single file")
        return [audio_path]
    
    print(f"📊 Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    # If audio is short enough, return as-is
    if total_duration <= chunk_duration + 30:  # 30 second buffer
        print("   Audio is short enough, no splitting needed")
        return [audio_path]
    
    # Setup output directory
    if output_dir is None:
        output_dir = str(Path(audio_path).parent)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of chunks
    effective_chunk = chunk_duration - overlap
    num_chunks = math.ceil(total_duration / effective_chunk)
    
    print(f"✂️ Splitting into {num_chunks} chunks ({chunk_duration}s each with {overlap}s overlap)")
    
    ffmpeg_path = get_ffmpeg_path()
    audio_name = Path(audio_path).stem
    chunk_paths = []
    
    for i in range(num_chunks):
        start_time = i * effective_chunk
        
        # Last chunk might be shorter
        if start_time >= total_duration:
            break
        
        chunk_path = os.path.join(output_dir, f"{audio_name}_chunk_{i+1:03d}.wav")
        
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            if result.returncode == 0 and os.path.exists(chunk_path):
                chunk_paths.append(chunk_path)
                print(f"   ✅ Chunk {i+1}/{num_chunks}: {start_time:.0f}s - {min(start_time + chunk_duration, total_duration):.0f}s")
            else:
                print(f"   ⚠️ Chunk {i+1} failed: {result.stderr[:100] if result.stderr else 'Unknown error'}")
                
        except Exception as e:
            print(f"   ❌ Chunk {i+1} error: {e}")
    
    if not chunk_paths:
        print("⚠️ No chunks created, falling back to original file")
        return [audio_path]
    
    print(f"✅ Created {len(chunk_paths)} audio chunks")
    return chunk_paths


def cleanup_chunks(chunk_paths: List[str]):
    """Remove temporary chunk files."""
    for path in chunk_paths:
        try:
            if os.path.exists(path) and "_chunk_" in path:
                os.remove(path)
        except Exception as e:
            print(f"⚠️ Could not remove chunk {path}: {e}")


def combine_transcriptions(
    transcriptions: List[str],
    overlap_words: int = 10
) -> str:
    """
    Combine transcriptions from multiple chunks.
    Handles overlapping text between chunks.
    
    Args:
        transcriptions: List of transcribed texts from each chunk
        overlap_words: Number of words to check for overlap
        
    Returns:
        Combined text
    """
    if not transcriptions:
        return ""
    
    if len(transcriptions) == 1:
        return transcriptions[0]
    
    combined = transcriptions[0]
    
    for i in range(1, len(transcriptions)):
        current = transcriptions[i]
        
        if not current.strip():
            continue
        
        # Try to find overlap
        combined_words = combined.split()
        current_words = current.split()
        
        if len(combined_words) < overlap_words or len(current_words) < overlap_words:
            # Not enough words to check overlap, just append
            combined = combined.rstrip() + " " + current.lstrip()
            continue
        
        # Look for overlap in the last N words of combined and first N words of current
        overlap_found = False
        for j in range(min(overlap_words, len(combined_words))):
            # Get last j+1 words from combined
            last_words = " ".join(combined_words[-(j+1):]).lower()
            
            # Check if they appear at the start of current
            for k in range(min(overlap_words, len(current_words))):
                first_words = " ".join(current_words[:k+1]).lower()
                
                if last_words == first_words:
                    # Found overlap, skip the overlapping part
                    combined = combined.rstrip() + " " + " ".join(current_words[k+1:])
                    overlap_found = True
                    break
            
            if overlap_found:
                break
        
        if not overlap_found:
            # No overlap found, just append with space
            combined = combined.rstrip() + " " + current.lstrip()
    
    # Clean up extra spaces
    combined = " ".join(combined.split())
    
    return combined


class AudioChunker:
    """
    High-level class for chunked audio processing.
    """
    
    def __init__(
        self,
        chunk_duration: int = 180,  # 3 minutes
        overlap: int = 5,  # 5 seconds
        cleanup_after: bool = True
    ):
        """
        Initialize the audio chunker.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            cleanup_after: Whether to delete chunk files after processing
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.cleanup_after = cleanup_after
    
    def process_with_callback(
        self,
        audio_path: str,
        process_func,  # Function that takes audio_path and returns transcription
        output_dir: str = None
    ) -> Tuple[str, List[str]]:
        """
        Process audio file in chunks using a callback function.
        
        Args:
            audio_path: Path to the audio file
            process_func: Function to process each chunk (takes path, returns text)
            output_dir: Directory for temporary chunks
            
        Returns:
            Tuple of (combined_text, list_of_chunk_texts)
        """
        # Split audio into chunks
        chunk_paths = split_audio_into_chunks(
            audio_path,
            output_dir=output_dir,
            chunk_duration=self.chunk_duration,
            overlap=self.overlap
        )
        
        # Process each chunk
        transcriptions = []
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n🎤 Processing chunk {i+1}/{len(chunk_paths)}...")
            try:
                text = process_func(chunk_path)
                transcriptions.append(text)
                print(f"   ✅ Got {len(text)} characters")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                transcriptions.append("")
        
        # Combine transcriptions
        print("\n🔗 Combining transcriptions...")
        combined = combine_transcriptions(transcriptions)
        
        # Cleanup
        if self.cleanup_after and chunk_paths[0] != audio_path:
            cleanup_chunks(chunk_paths)
            print("🗑️ Cleaned up temporary chunks")
        
        return combined, transcriptions


if __name__ == "__main__":
    # Test the chunker
    print("Audio Chunker Module")
    print("=" * 50)
    
    # Test with a sample file
    test_audio = "test_audio.wav"
    if os.path.exists(test_audio):
        duration = get_audio_duration(test_audio)
        print(f"Test file duration: {duration}s")
        
        chunks = split_audio_into_chunks(test_audio, chunk_duration=60)
        print(f"Created {len(chunks)} chunks")
        
        # Cleanup
        cleanup_chunks(chunks)
    else:
        print("No test audio file found")
