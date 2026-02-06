@echo off
REM Fast CPU-optimized pipeline for attendy.mp4
REM Uses faster model and reduced refinement passes

echo ========================================
echo Fast CPU-Optimized Pipeline
echo ========================================

python scripts/run_pipeline.py --video attendy.mp4 --out output --fast

echo.
echo ========================================
echo Processing complete!
echo Check output folder for results
echo ========================================
pause
