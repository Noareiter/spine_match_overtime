@echo off
setlocal
cd /d "%~dp0"

echo Starting spine match overtime annotator ...
echo Config: config\annotator.json
echo.

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul

set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY where python >nul 2>&1 && set PY=python
if not defined PY (
  echo ERROR: Python not found.
  pause
  exit /b 1
)

%PY% -u scripts\run_annotator.py --config config\annotator.json --open-folders
set ERR=%ERRORLEVEL%
echo.
if %ERR% neq 0 (echo Server exited with error %ERR%.) else (echo Server stopped.)
pause
exit /b %ERR%
