@echo off
rem Upgrades the hand firmware with Wuji CLI and re-probes the fingertips. Keep the hand powered and on Ethernet.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0wuji_upgrade.ps1"
echo.
echo [done] report in runs\tactile\wuji_upgrade_*.txt
pause
