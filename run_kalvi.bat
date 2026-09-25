@echo off
rem Start Kalvi and open it in the browser.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m kalvi %*
) else (
  python -m kalvi %*
)
