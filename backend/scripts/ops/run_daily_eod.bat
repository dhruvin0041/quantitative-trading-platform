@echo off
REM ==============================================================================
REM Manual On-Demand Daily Execution Runner (Windows)
REM
REM Strictly manual and on-demand execution. No background daemons or loops.
REM Executes a single daily execution cycle and terminates immediately.
REM
REM Usage:
REM   run_daily_eod.bat              (Live paper execution with Pure XGBoost flagship)
REM   run_daily_eod.bat --dry-run    (Simulated run with zero broker/db mutations)
REM   run_daily_eod.bat --use-veto   (Enables secondary asymmetric veto consensus)
REM   run_daily_eod.bat --status     (Read-only status dashboard: balance, positions, stops)
REM ==============================================================================

:: Navigate to project root explicitly
cd /d "D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator"

:: Ensure artifacts directory exists for logging
if not exist "backend\artifacts" mkdir "backend\artifacts"

:: Resolve virtual environment Python executable
set PYTHON_BIN=backend\venv\Scripts\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=venv\Scripts\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=python.exe

echo [%DATE% %TIME%] Starting Manual On-Demand Daily Execution Run... >> backend\artifacts\paper_execution.log 2>&1

"%PYTHON_BIN%" -m backend.execution.paper_runner %* >> backend\artifacts\paper_execution.log 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Manual On-Demand Daily Execution Run finished with exit code %EXIT_CODE%. >> backend\artifacts\paper_execution.log 2>&1

exit /b %EXIT_CODE%
