@echo off
rem Smoothest-motion showcase on the real Wuji Hand 2: minimum-jerk trajectories, 250 Hz commands with velocity
rem feedforward, kp 5.0 / kd 0.1, 1.5 A, firmware friction feed-forward. 8 s countdown - keep clear of the hand.
rem Ctrl+C in this window stops it (the hand eases back and the motors switch off).
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
if not exist "%REPO%\.venv-deploy\Scripts\python.exe" (
  echo [setup] run tactile_viewer.bat once first to create the environment
  pause
  exit /b 1
)
"%REPO%\.venv-deploy\Scripts\python.exe" "%REPO%\deploy\smooth_demo.py" %*
echo.
echo [done] exit code %ERRORLEVEL% - log in runs\real_hand\
pause
endlocal
