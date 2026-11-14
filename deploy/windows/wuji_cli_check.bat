@echo off
rem Installs Wuji CLI (official installer) if needed and runs its hand/tactile diagnostics; no firmware is flashed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0wuji_cli_check.ps1"
echo.
echo [done] report in runs\tactile\wuji_cli_*.txt
pause
