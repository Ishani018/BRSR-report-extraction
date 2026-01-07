@echo off
echo Installing PDF Processing Pipeline Requirements...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Create virtual environment (optional but recommended)
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing Python packages...
pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo IMPORTANT: You also need to install Tesseract OCR
echo Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo Or install via chocolatey: choco install tesseract
echo.
echo To run the pipeline:
echo 1. Place PDF files in the 'data' directory
echo 2. Run: python main.py
echo.
pause
