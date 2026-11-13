@echo off
rem Writes a diagnostic report about the fingertip tactile modules to runs\tactile\probe_<time>.txt
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "VENV=%REPO%\.venv-deploy"
if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] run tactile_viewer.bat once first to create the environment
  pause
  exit /b 1
)
"%VENV%\Scripts\python.exe" "%REPO%\deploy\tactile_probe.py" %*
echo.
echo [done] exit code %ERRORLEVEL% - report in runs\tactile\
pause
endlocal
