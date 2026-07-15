@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call SETUP_ONCE.bat
if errorlevel 1 exit /b 1
set /p SOURCE=Stem folder:
set /p OUTPUT=Clean output folder:
".venv\Scripts\python.exe" -m stem_bleed_reclaimer.cli "%SOURCE%" "%OUTPUT%"
pause
