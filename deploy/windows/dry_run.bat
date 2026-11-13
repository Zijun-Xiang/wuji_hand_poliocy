@echo off
rem Dry run: Wuji stub hand, no hardware touched. Installs the environment on first use.
call "%~dp0play_loop.bat" --no-hardware --speed 4 --auto-start 3
pause
