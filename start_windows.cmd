@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.cmd first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set CUDA_VISIBLE_DEVICES=-1
".venv\Scripts\python.exe" scripts\verify_environment.py --offline
if errorlevel 1 (
  echo Demo checks failed. Run setup_windows.cmd while online to repair dependencies or missing assets.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
