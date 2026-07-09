@echo off
REM ============================================================
REM  One-click launcher: AI Footfall Analytics Dashboard
REM  Double-click this file to start the Streamlit dashboard.
REM ============================================================
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at .venv
    echo         Create it first ^(see README^): py -3.13 -m venv .venv
    echo.
    pause
    exit /b 1
)

echo Starting the AI Footfall Analytics Dashboard...
echo A browser tab will open automatically. Close this window to stop.
echo.
"%PY%" -m streamlit run app\dashboard.py

pause
