@echo off
rem Reboots the hand (joints must be disabled, hand goes limp ~15 s) and re-checks the fingertip modules.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
"%REPO%\.venv-deploy\Scripts\python.exe" "%REPO%\deploy\tactile_reboot_probe.py" %*
echo.
echo [done] exit code %ERRORLEVEL% - report in runs\tactile\
pause
endlocal
