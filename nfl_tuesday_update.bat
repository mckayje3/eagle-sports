@echo off
REM NFL Tuesday Update - Run every Tuesday morning after Monday Night Football
REM This script runs the full Tuesday update process

cd /d "%~dp0"
echo Starting NFL Tuesday Update at %date% %time%
echo.

REM Run the Tuesday update script
py nfl_tuesday_update.py

echo.
echo Update completed at %date% %time%
pause
