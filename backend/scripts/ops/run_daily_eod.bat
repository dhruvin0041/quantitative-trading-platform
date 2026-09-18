@echo off
REM ==============================================================================
REM Institutional Automated Daily Execution Runner (Windows Task Scheduler / Batch)
REM 
REM Cron Schedule Equivalent:
REM   Mon-Fri at 16:15 EST (15 minutes after US market close: 21:15 UTC standard)
REM   Cron Expression: 15 16 * * 1-5
REM 
REM Windows Task Scheduler Setup:
REM   schtasks /create /tn "StockIndicator_DailyEOD" /tr "%~dp0run_daily_eod.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:15
REM 
REM Usage:
REM   run_daily_eod.bat              (Live paper execution with Pure XGBoost flagship)
REM   run_daily_eod.bat --dry-run    (Simulated run with zero broker/db mutations)
REM   run_daily_eod.bat --use-veto   (Enables secondary asymmetric veto consensus)
REM ==============================================================================

setlocal enabledelayedexpansion

REM Set root working directory to backend
cd /d "%~dp0..\.."

REM Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

echo [%DATE% %TIME%] Starting Automated Daily EOD Execution Run...
python execution\paper_runner.py %*
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% equ 0 (
    echo [%DATE% %TIME%] Automated Daily EOD Execution Run completed successfully.
) else (
    echo [%DATE% %TIME%] ERROR: Daily EOD Execution Run failed with exit code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
