#!/bin/bash
# Kanji-Kakei Startup Script for Git Bash / MINGW64

# Move to the script's directory
cd "$(dirname "$0")"

# Execute python in the virtual environment
./.venv/Scripts/python main.py
