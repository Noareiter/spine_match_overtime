@echo off
cd /d "%~dp0"
echo Building results/ folder skeleton from build_results_config.txt ...
echo.
python build_results_skeleton.py
echo.
pause
