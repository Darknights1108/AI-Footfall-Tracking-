@echo off
REM ============================================================
REM  AI Footfall Tracking & Retail Analytics - control panel
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

:menu
cls
echo ============================================================
echo    AI Footfall Tracking ^& Retail Analytics
echo ============================================================
echo.
echo   [1]  Launch dashboard (Streamlit)
echo   [2]  Process a MOT17 sequence (with annotated video)
echo   [3]  Reset the database
echo   [4]  Exit
echo.
set /p "choice=Select an option [1-4]: "

if "%choice%"=="1" goto dashboard
if "%choice%"=="2" goto process
if "%choice%"=="3" goto reset
if "%choice%"=="4" goto end
echo Invalid choice.
timeout /t 1 >nul
goto menu

:dashboard
echo.
echo Starting dashboard... a browser tab will open. Close this window to stop.
echo.
"%PY%" -m streamlit run app\dashboard.py
goto menu

:process
echo.
echo Recommended demo sequences (line orientation matters):
echo    MOT17-09-FRCNN  -^> horizontal   (richest demo)
echo    MOT17-02-FRCNN  -^> horizontal
echo    MOT17-04-FRCNN  -^> vertical
echo.
set "seq=MOT17-09-FRCNN"
set /p "seq=Sequence name [%seq%]: "
set "line=horizontal"
set /p "line=Line orientation (horizontal/vertical) [%line%]: "
echo.
echo Processing %seq% with a %line% line...
echo.
"%PY%" scripts\process_mot17.py --sequence "MOT17\train\%seq%" --line %line% --export-video
echo.
pause
goto menu

:reset
echo.
set /p "confirm=This will DELETE all footfall data. Continue? [y/N]: "
if /i "%confirm%"=="y" (
    "%PY%" scripts\reset_db.py
) else (
    echo Cancelled.
)
echo.
pause
goto menu

:end
echo Bye.
