@echo off
cd /d "%~dp0"

REM Use the venv python from the SAME directory tree
set "VENV_PYTHON=%~dp0..\..\..\venv311\Scripts\python.exe"

REM Check if venv python exists
if not exist "%VENV_PYTHON%" (
    echo Venv python not found at %VENV_PYTHON%
    echo Trying system python...
    set "VENV_PYTHON=python"
)

echo Using Python: %VENV_PYTHON%

REM Install required packages
"%VENV_PYTHON%" -m pip install fastapi uvicorn python-multipart 2>nul

REM Start the server
"%VENV_PYTHON%" -m uvicorn server:app --host 0.0.0.0 --port 8000
