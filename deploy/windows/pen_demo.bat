@echo off
rem Pen-spinning demo on the real hand (joint playback of the stitched loop, no camera):
rem the hand eases to the grip pose, then counts down 25 s while you place the pen between
rem the thumb, index and middle fingertips, then plays 2 laps at half speed. Ctrl+C = ease open + stop.
call "%~dp0play_loop.bat" --laps 2 --speed 0.5 --hold-s 2 --auto-start 25 %*
