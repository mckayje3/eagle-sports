@echo off
REM Daily Basketball Update - Run via Task Scheduler
REM Recommended schedule: Every day at 10:00 AM and 6:00 PM ET

cd /d C:\Users\jbeast\documents\coding\sports
C:\Users\jbeast\anaconda3\python.exe daily_basketball_update.py

REM Keep window open if there's an error
if %ERRORLEVEL% NEQ 0 pause
