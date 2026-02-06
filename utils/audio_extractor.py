"""
Audio Extraction Module
Extracts audio from video files using FFmpeg
Supports long videos (10+ minutes)
"""

import subprocess
import os
import sys
import json
from pathlib import Path
from typing import Optional


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
    
    raise FileNotFoundError(
        "FFmpeg not found! Install with: winget install Gyan.FFmpeg\n"
        "Then restart your terminal/VS Code."
    )


def get_ffprobe_path() -> str:
    """Find FFprobe executable path."""
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path == "ffmpeg":
        return "ffprobe"
    return ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")


def extract_audio(video_path: str, audio_path: str = None) -> str:
    """
    Extract audio from a video file using subprocess.
    Supports long videos (10+ minutes).
    
    Args:
        video_path: Path to the input video file
        audio_path: Path for the output audio file (optional)
        
    Returns:
        Path to the extracted audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if audio_path is None:
        video_name = Path(video_path).stem
        audio_path = str(Path(video_path).parent / f"{video_name}_audio.wav")
    
    output_dir = os.path.dirname(audio_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    ffmpeg_path = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    
    try:
        print(f"🎵 Extracting audio from: {os.path.basename(video_path)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg error: {error_msg}")
        
        if not os.path.exists(audio_path):
            raise RuntimeError("Audio extraction failed - output file not created")
        
        print(f"✅ Audio extracted: {audio_path}")
        return audio_path
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "FFmpeg not found! Install with: winget install Gyan.FFmpeg\n"
            "Then restart your terminal/VS Code."
        )
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {str(e)}")


def get_audio_info(audio_path: str) -> dict:
    """Get information about an audio file."""
    if not os.path.exists(audio_path):
        return {'duration': 0}
    
    try:
        ffprobe_path = get_ffprobe_path()
        
        cmd = [
            ffprobe_path, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            audio_path
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data.get('format', {}).get('duration', 0))
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    return {
                        'duration': duration,
                        'duration_formatted': format_duration(duration),
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0)),
                        'codec': stream.get('codec_name', 'unknown')
                    }
        return {'duration': 0}
        
    except Exception as e:
        print(f"Warning: Could not get audio info: {e}")
        return {'duration': 0}


def format_duration(seconds: float) -> str:
    """Format duration to MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    print("Audio Extraction Module")
    try:
        print(f"FFmpeg path: {get_ffmpeg_path()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
