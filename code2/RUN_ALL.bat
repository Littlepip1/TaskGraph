@echo off
cd /d "%~dp0"
python run_dynamic_demand.py --mode full --workers 6
if errorlevel 1 (
  echo Stopped. Inspect the error above; rerun the same command to resume.
  pause
  exit /b 1
)
echo Return runs\full\DYNAMIC_DEMAND_RETURN.zip
pause
