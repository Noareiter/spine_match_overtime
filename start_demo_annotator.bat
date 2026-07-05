@echo off
setlocal
cd /d "%~dp0"

echo Spine match overtime — DEMO mode
echo.

set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY where python >nul 2>&1 && set PY=python
if not defined PY (
  echo ERROR: Python not found.
  pause
  exit /b 1
)

echo [1/2] Building demo data...
%PY% scripts\create_spine_logic_demo.py
if errorlevel 1 (
  echo Demo data generation failed.
  pause
  exit /b 1
)

echo.
echo [2/2] Starting server on http://127.0.0.1:8010
echo   Dendrite linker: http://127.0.0.1:8010/mtp/
echo   Spine tracker:   http://127.0.0.1:8010/mtp/viewer/?fov=1
echo.

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul

%PY% -u scripts\run_annotator.py --config config\annotator_demo.json --open-folders
set ERR=%ERRORLEVEL%
echo.
if %ERR% neq 0 (echo Server exited with error %ERR%.) else (echo Server stopped.)
pause
exit /b %ERR%
