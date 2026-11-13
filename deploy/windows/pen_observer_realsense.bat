@echo off
rem Wuji's official pen observer fed by an Intel RealSense colour stream (needs the tagged pen + wrist tag).
rem   deploy\windows\pen_observer_realsense.bat            -> 1280x720 @ 30 fps with preview window
rem   deploy\windows\pen_observer_realsense.bat --width 848 --height 480 --fps 60
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "VENV=%REPO%\.venv-deploy"
set "WUJI=%REPO%\.deps\wuji-mjlab"
if exist "%REPO%\..\wuji-mjlab\deploy\pen_spin\runtime\hand.py" for %%I in ("%REPO%\..\wuji-mjlab") do set "WUJI=%%~fI"
if not exist "%VENV%\Scripts\python.exe" (
  echo run deploy\windows\dry_run.bat first: it creates the Python environment
  exit /b 1
)
"%VENV%\Scripts\python.exe" -c "import cv2, zmq, pyrealsense2" >nul 2>&1 || (
  echo [setup] installing opencv-contrib-python, pyzmq, pyrealsense2
  "%VENV%\Scripts\python.exe" -m pip install opencv-contrib-python pyzmq pyrealsense2 || exit /b 1
)
"%VENV%\Scripts\python.exe" "%REPO%\deploy\pen_observer_realsense.py" --wuji-root "%WUJI%" --preview %*
endlocal
