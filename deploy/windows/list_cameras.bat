@echo off
rem Lists the cameras and imaging USB devices Windows currently sees, into deploy\windows\cameras.txt
powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in @('Camera','Image','USB','MEDIA') } | Select-Object Status,Class,FriendlyName,InstanceId | Format-Table -AutoSize | Out-String -Width 300 | Out-File -Encoding utf8 '%~dp0cameras.txt'"
type "%~dp0cameras.txt"
pause
