@echo off
setlocal
cd /d "%~dp0"
set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY where python >nul 2>&1 && set PY=python
if not defined PY (
  echo ERROR: Python not found.
  pause
  exit /b 1
)
%PY% scripts\open_animal_folders.py
pause
