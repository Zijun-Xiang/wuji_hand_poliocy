@echo off
rem Which joints/buses of the Wuji Hand 2 are online, fault codes, and (optional) a soft reboot.
rem   deploy\windows\joint_probe.bat                 -> read-only snapshot
rem   deploy\windows\joint_probe.bat --clear-fault   -> also clear latched faults
rem   deploy\windows\joint_probe.bat --reboot        -> also soft-reboot the hand (goes limp ~15 s) and re-check
rem Writes runs\real_hand\joint_probe_<time>.txt. Motors stay disabled.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "PY=%REPO%\.venv-deploy\Scripts\python.exe"
if not exist "%PY%" (
  echo [setup] run deploy\windows\play_loop.bat --no-hardware once to create .venv-deploy
  pause
  exit /b 1
)
set PYTHONUNBUFFERED=1
"%PY%" "%REPO%\deploy\joint_probe.py" %*
pause
endlocal
