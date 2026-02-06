# 🎬 AI Video to Fluent Text

> A multi-modal AI system that converts spoken English from videos into fluent, professional text using transformer-based neural networks.

## 🎯 Project Overview

This system:
1. **Extracts audio** from uploaded videos
2. **Converts speech to text** using Whisper (Neural Network)
3. **Improves English** (grammar, fluency, professionalism)
4. **Displays** the final polished script

## 🏗️ Architecture

```
Video Upload → Audio Extraction → Speech-to-Text → Text Improvement → Fluent Output
     │              │                   │                 │              │
   FFmpeg        FFmpeg            Whisper AI      Transformer      Final Text
                                   (OpenAI)         Model
```

## 📂 Project Structure

```
ai-video-to-fluent-text/
├── backend/
│   ├── main.py              # FastAPI server
│   └── pipeline.py          # Complete AI pipeline
├── frontend/
│   ├── index.html           # Web UI
│   ├── styles.css           # Styling
│   └── script.js            # Frontend logic
├── utils/
│   ├── audio_extractor.py   # FFmpeg audio extraction
│   ├── speech_to_text.py    # Whisper transcription
│   ├── text_cleaner.py      # Text preprocessing
│   └── text_improver.py     # Grammar correction
├── tests/
│   └── test_pipeline.py     # Test scripts
├── models/                  # Cached AI models
├── datasets/                # Training/test data
├── uploads/                 # Temporary files
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

- **Speech-to-Text**: OpenAI Whisper
- **Text Improvement**: Transformer-based grammar correction
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Audio Processing**: FFmpeg

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the backend
cd backend
uvicorn main:app --reload

# 4. Open frontend
# Open frontend/index.html in browser
```

## 💻 Local Execution (CLI)

For quick local testing without the web interface:

```bash
# 1. Run the automatic local script (Windows)
./run_local.bat

# 2. Or run manually via Python
python scripts/run_pipeline.py --video attendy.mp4 --out output
```

## 📊 Neural Networks Used

1. **Whisper** - Transformer-based ASR (Automatic Speech Recognition)
2. **T5/Grammar Correction Model** - Text-to-Text Transformer for fluency

## 👨‍💻 Author
Salman Malhi
