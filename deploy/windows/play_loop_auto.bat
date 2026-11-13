@echo off
rem Real hand, one lap at half speed, starts by itself after a 10 s countdown (Ctrl+C eases the hand open and stops).
call "%~dp0play_loop.bat" --speed 0.5 --auto-start 10
pause
