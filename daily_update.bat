@echo off
REM ============================================================
REM DAILY SPORTS DATA UPDATE
REM Automatically detects which sports are in season and updates
REM only those. Fetches odds, updates predictions, pushes to cloud.
REM
REM Schedule with Windows Task Scheduler to run daily at 9 AM
REM ============================================================

cd /d C:\Users\jbeast\documents\coding\sports

REM Activate conda environment if needed (uncomment if using conda)
REM call C:\Users\jbeast\anaconda3\Scripts\activate.bat base

REM Run the smart daily update script
python daily_update.py %*

REM Keep window open if run manually (comment out for scheduled task)
REM pause
