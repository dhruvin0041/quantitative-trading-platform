@echo off
REM ==============================================================================
REM Institutional Automated Daily Execution Runner (Windows Task Scheduler / Batch)
REM 
REM Windows Task Scheduler Setup:
REM   schtasks /create /tn "StockIndicator_DailyEOD" /tr "D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator\backend\scripts\ops\run_daily_eod.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:15 /f
REM ==============================================================================

:: Navigate to project root explicitly
cd /d "D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator"

:: Ensure artifacts directory exists for logging
if not exist "backend\artifacts" mkdir "backend\artifacts"

:: Resolve virtual environment Python executable
set PYTHON_BIN=backend\venv\Scripts\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=venv\Scripts\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=python.exe

echo [%DATE% %TIME%] Starting Automated Daily EOD Execution Run... >> backend\artifacts\paper_execution.log 2>&1

"%PYTHON_BIN%" -m backend.execution.paper_runner %* >> backend\artifacts\paper_execution.log 2>&1

echo [%DATE% %TIME%] Automated Daily EOD Execution Run finished. >> backend\artifacts\paper_execution.log 2>&1
