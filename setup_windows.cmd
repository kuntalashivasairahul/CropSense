@echo off
setlocal
cd /d "%~dp0"
py -3.12 scripts\setup_environment.py
if errorlevel 1 (
  echo Setup failed. Read the message above. Python 3.12 64-bit and an internet connection are required.
  pause
  exit /b 1
)
pause
