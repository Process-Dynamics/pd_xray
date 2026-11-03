@echo off
setlocal
cd /d "%~dp0"
set "VIEWER_PYTHON="
if exist "%~dp0..\..\.venv\Scripts\python.exe" set "VIEWER_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not defined VIEWER_PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "VIEWER_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined VIEWER_PYTHON if exist "%USERPROFILE%\anaconda3\python.exe" set "VIEWER_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not defined VIEWER_PYTHON set "VIEWER_PYTHON=python"
"%VIEWER_PYTHON%" "%~dp0access_phase_viewer.py" --open-browser %*
if errorlevel 1 (
  echo Viewer stopped with an error.
  pause
)
