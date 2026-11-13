@echo off
rem Camera-less playback of a stitched loop on the real Wuji Hand 2, from Windows.
rem   deploy\windows\play_loop.bat                      -> one lap, half speed (first try)
rem   deploy\windows\play_loop.bat --speed 1.0 --laps 3
rem   deploy\windows\play_loop.bat --no-hardware        -> dry run with Wuji's stub hand
rem Everything is installed inside this repository (.venv-deploy, .deps\wuji-mjlab).
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "VENV=%REPO%\.venv-deploy"
set "WUJI=%REPO%\.deps\wuji-mjlab"

if exist "%VENV%\Scripts\python.exe" goto :have_venv
echo [setup] creating %VENV%
set "PY="
for %%V in (3.13 3.12 3.11 3.10 3.14) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
)
if not defined PY ( python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo [setup] no Python 3.10-3.14 found. Install Python from python.org, then run this again.
  exit /b 1
)
%PY% -m venv "%VENV%" || exit /b 1
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
echo [setup] installing numpy, pyyaml, wuji_sdk==2026.11.9 (Wuji Hand 2 driver, Windows wheel)
"%VENV%\Scripts\python.exe" -m pip install numpy pyyaml "wuji_sdk==2026.11.9" || exit /b 1
:have_venv

if exist "%WUJI%\deploy\pen_spin\runtime\hand.py" goto :have_wuji
if exist "%REPO%\..\wuji-mjlab\deploy\pen_spin\runtime\hand.py" (
  for %%I in ("%REPO%\..\wuji-mjlab") do set "WUJI=%%~fI"
  goto :have_wuji
)
echo [setup] cloning wuji-mjlab (Wuji's deploy runtime) into %WUJI%
git clone --depth 1 --branch v2026.10.22 https://github.com/wuji-technology/wuji-mjlab "%WUJI%" || exit /b 1
:have_wuji

echo [run] wuji-mjlab: %WUJI%
if "%~1"=="" (
  "%VENV%\Scripts\python.exe" "%REPO%\deploy\play_reference.py" --wuji-root "%WUJI%" --motion "%REPO%\deploy\clips\graph_loop_1lap.npz" --speed 0.5 --log "%REPO%\deploy\last_playback.npz"
) else (
  "%VENV%\Scripts\python.exe" "%REPO%\deploy\play_reference.py" --wuji-root "%WUJI%" --motion "%REPO%\deploy\clips\graph_loop_1lap.npz" --log "%REPO%\deploy\last_playback.npz" %*
)
endlocal
