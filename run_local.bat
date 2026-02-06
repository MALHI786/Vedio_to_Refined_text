@echo off
echo ==========================================
echo 🎬 Running AI Video to Fluent Text Locally
echo ==========================================

:: Check if virtual environment exists
if not exist .venv\Scripts\activate.bat (
    if not exist venv\Scripts\activate.bat (
        echo ❌ Virtual environment not found! 
        echo Please run: python -m venv venv
        echo Then: pip install -r requirements.txt
        pause
        exit /b
    ) else (
        set VENV_PATH=venv\Scripts\activate.bat
    )
) else (
    set VENV_PATH=.venv\Scripts\activate.bat
)

:: Activate venv
echo 🔄 Activating virtual environment...
call %VENV_PATH%

:: Create output directory
if not exist local_output mkdir local_output

:: Run the pipeline
echo 🚀 Starting AI Pipeline on attendy.mp4...
python scripts/run_pipeline.py --video attendy.mp4 --out local_output

echo.
echo ==========================================
echo ✅ Execution Complete! 
echo Check the 'local_output' folder for results.
echo ==========================================
pause
