@echo off
cd /d "%~dp0"
python -u run_all_experiments.py --resume %*
if errorlevel 1 (
  echo Execution failed. Inspect the terminal and .error.json files. Run again to resume.
  exit /b 1
)

