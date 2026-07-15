@echo off
setlocal
cd /d "%~dp0"
set /p SOURCE=Stem folder: 
set /p OUTPUT=Clean output folder: 
python -m pip install -e . >nul
python -m stem_bleed_reclaimer.cli "%SOURCE%" "%OUTPUT%"
pause

