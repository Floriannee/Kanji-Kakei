@echo off
cd /d "%~dp0"
title Kanji-Kakei Startup Launcher
echo =========================================================
echo  Kanji-Kakei: Japanese Receipt Reader ^& AI Advisor
echo  Starting Windows 11 Desktop Application...
echo =========================================================
echo.

:: Check virtual environment
if not exist ".venv" (
    echo [ERROR] Virtual environment .venv was not found in this folder.
    echo Please run setup first.
    pause
    exit /b
)

:: Run application
echo Launching GUI main loop...
.venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Application exited with error code %errorlevel%.
    pause
)
