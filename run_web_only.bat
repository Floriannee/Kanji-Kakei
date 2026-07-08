@echo off
cd /d "%~dp0"
title Kanji-Kakei Web-Only Launcher
echo =========================================================
echo  Kanji-Kakei: Japanese Receipt Reader ^& AI Advisor (Headless)
echo  Starting Web-Only Dashboard...
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
echo Launching headless server...
.venv\Scripts\python.exe main.py --web-only
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Application exited with error code %errorlevel%.
    pause
)
