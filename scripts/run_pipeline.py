#!/usr/bin/env python3
"""
Simple CLI to run the AI pipeline on a video, audio, or raw text file.

Usage examples:
  python scripts/run_pipeline.py --video path/to/video.mp4
  python scripts/run_pipeline.py --audio path/to/audio.wav
  python scripts/run_pipeline.py --text "raw transcription text"

Outputs saved to `output/` by default.
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', help='Path to input video file')
    group.add_argument('--audio', help='Path to input audio file')
    group.add_argument('--text', help='Raw text to refine')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--whisper', default='base', help='Whisper model (tiny, base, small, medium, large)')
    parser.add_argument('--grammar', default='vennify/t5-base-grammar-correction', help='Grammar model name')
    parser.add_argument('--fast', action='store_true', help='Use fast CPU-optimized settings')
    parser.add_argument('--passes', type=int, default=1, help='Number of refinement passes (1-3)')
    parser.add_argument('--keep-audio', action='store_true', help='Keep extracted audio')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from backend.pipeline import AIPipeline

    # Use fast model if requested
    if args.fast:
        grammar_model = 'prithivida/grammar_error_correcter_v1'
        passes = 1
        print("⚡ Fast mode enabled: Using CPU-optimized model")
    else:
        grammar_model = args.grammar
        passes = args.passes
    
    pipeline = AIPipeline(
        whisper_model=args.whisper, 
        grammar_model=grammar_model,
        refinement_passes=passes,
        low_resource=True  # Enable CPU optimizations
    )

    if args.video:
        res = pipeline.process_video(args.video, output_dir=args.out, keep_audio=args.keep_audio)
    elif args.audio:
        res = pipeline.process_audio(args.audio)
    else:
        res = pipeline.process_text(args.text, for_script=True)

    # Save outputs
    improved_path = os.path.join(args.out, 'improved_script.txt')
    script_path = os.path.join(args.out, 'voiceover_script.txt')

    with open(improved_path, 'w', encoding='utf-8') as f:
        f.write(res.get('improved_text', res.get('cleaned_text', '')))

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(res.get('script_text', res.get('improved_text', '')))

    print('Saved improved script to', improved_path)
    print('Saved voiceover script to', script_path)


if __name__ == '__main__':
    main()
