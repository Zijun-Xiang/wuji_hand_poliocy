@echo off
rem Live viewer for the Wuji Hand 2 fingertip tactile sensors (hand on power + Ethernet/USB).
rem   deploy\windows\tactile_viewer.bat                 -> live window
rem   deploy\windows\tactile_viewer.bat --no-hardware   -> synthetic data (GUI check)
rem   deploy\windows\tactile_viewer.bat --dump-only     -> print each sensor's format + raw frames
rem Uses the same .venv-deploy as play_loop.bat (created here if missing) plus matplotlib.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "VENV=%REPO%\.venv-deploy"

if exist "%VENV%\Scripts\python.exe" goto :have_venv
echo [setup] creating %VENV%
set "PYEXE="
set "PYCMD="
for %%V in (313 312 311 310 314) do (
  if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
)
if defined PYEXE goto :make_venv
for %%V in (3.13 3.12 3.11 3.10 3.14) do (
  if not defined PYCMD ( py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V" )
)
if not defined PYCMD ( python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD (
  echo [setup] no Python 3.10-3.14 found. Install Python from python.org, then run this again.
  pause
  exit /b 1
)
:make_venv
if defined PYEXE (
  echo [setup] using %PYEXE%
  "%PYEXE%" -m venv "%VENV%"
) else (
  echo [setup] using %PYCMD%
  %PYCMD% -m venv "%VENV%"
)
if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] could not create the virtual environment
  pause
  exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
echo [setup] installing numpy, pyyaml, wuji_sdk==2026.11.9
"%VENV%\Scripts\python.exe" -m pip install numpy pyyaml "wuji_sdk==2026.11.9" || (pause & exit /b 1)
:have_venv

"%VENV%\Scripts\python.exe" -c "import matplotlib" >nul 2>&1 || (
  echo [setup] installing matplotlib
  "%VENV%\Scripts\python.exe" -m pip install matplotlib || (pause & exit /b 1)
)

set "WUJI=%REPO%\.deps\wuji-mjlab"
if exist "%WUJI%\deploy\pen_spin\runtime\hand.py" goto :have_wuji
if exist "%REPO%\..\wuji-mjlab\deploy\pen_spin\runtime\hand.py" (
  for %%I in ("%REPO%\..\wuji-mjlab") do set "WUJI=%%~fI"
  goto :have_wuji
)
echo [setup] cloning wuji-mjlab (Wuji's deploy runtime, needed for --home) into %WUJI%
git clone --depth 1 --branch v2026.10.22 https://github.com/wuji-technology/wuji-mjlab "%WUJI%" || (pause & exit /b 1)
:have_wuji

echo [run] tactile viewer  (close the window or press q to stop; the log goes to runs\tactile\)
"%VENV%\Scripts\python.exe" "%REPO%\deploy\tactile_viewer.py" --wuji-root "%WUJI%" %*
echo.
echo [done] exit code %ERRORLEVEL%
pause
endlocal
