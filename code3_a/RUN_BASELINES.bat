@echo off
cd /d "%~dp0"
python -u run_baseline_comparison.py --resume %*
if errorlevel 1 exit /b 1

