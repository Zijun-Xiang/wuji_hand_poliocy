@echo off
rem Pen-spinning loop on the real hand at FULL speed, with the RealSense recording the run:
rem   camera_record.py runs in the background for 150 s (mp4 + a snapshot every 3 s in runs\real_hand\),
rem   then pen_demo: grip pose -> 25 s pen-placement countdown -> 2 laps at speed 1.0. Ctrl+C = open + stop.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "PY=%REPO%\.venv-deploy\Scripts\python.exe"
"%PY%" -c "import pyrealsense2, cv2" >nul 2>&1 || (
  echo [setup] installing pyrealsense2 + opencv-python for the recording
  "%PY%" -m pip install pyrealsense2 opencv-python
)
echo [cam] recording to runs\real_hand\cam_^<time^>.mp4 (background, 150 s)
start "camera" /B /D "%REPO%" "%PY%" "%REPO%\deploy\camera_record.py" --seconds 150 --snap-every 3
timeout /t 3 /nobreak >nul
call "%~dp0play_loop.bat" --laps 2 --speed 1.0 --hold-s 2 --auto-start 25 %*
echo.
echo [done] camera files in runs\real_hand\cam_*
pause
endlocal
