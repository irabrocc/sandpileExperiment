@echo off
chcp 65001 >nul
echo ========================================
echo   Sandpile Project - Environment Setup
echo ========================================
echo/

:: Check if venv exists AND works (not broken from another machine)
set VENV_OK=0
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 (
        set VENV_OK=1
    ) else (
        echo [!] Virtual environment is broken - points to missing Python.
        echo [*] Removing broken venv and recreating...
        rmdir /s /q ".venv"
    )
)

:: Use call trick to expand VENV_OK at runtime without delayedexpansion
call :check_venv
goto :install

:check_venv
if "%VENV_OK%"=="1" (
    echo [OK] Virtual environment already exists.
    goto :eof
)
echo [*] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [!] Failed to create venv. Is Python installed and in PATH?
    pause
    exit /b 1
)
echo [OK] Virtual environment created.
goto :eof

:install
echo [*] Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [!] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo/
echo ========================================
echo   Setup complete! You're good to go.
echo ========================================
pause
