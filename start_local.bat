@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_local.ps1" %*
set "ROOP_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%ROOP_LAUNCH_EXIT%"=="0" pause
exit /b %ROOP_LAUNCH_EXIT%
