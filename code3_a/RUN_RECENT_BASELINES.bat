@echo off
cd /d "%~dp0"
python -u run_recent_paper_baselines.py --resume %*
if errorlevel 1 exit /b 1

