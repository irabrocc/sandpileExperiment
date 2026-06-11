@echo off
chcp 65001 >nul
echo ========================================
echo   Sandpile Project - Environment Setup
echo ========================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [✓] Virtual environment already exists.
) else (
    echo [*] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Failed to create venv. Is Python installed and in PATH?
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created.
)

echo [*] Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [!] Failed to install dependencies.
    pause
    exit /b 1
)
echo [✓] Dependencies installed.
echo.
echo ========================================
echo   Setup complete! You're good to go.
echo ========================================
pause
