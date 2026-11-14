@echo off
rem Moves the Wuji Hand 2 to its home pose (all joints 0 rad, open hand), holds it, and shows the
rem live fingertip tactile viewer. Touch the fingertips while it runs. q or closing the window
rem releases the hand (joints disabled). Keep fingers clear of the hand while it moves.
call "%~dp0tactile_viewer.bat" --home %*
