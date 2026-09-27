@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_local.ps1" %*
set "ROOP_STOP_EXIT=%ERRORLEVEL%"
if not "%ROOP_STOP_EXIT%"=="0" pause
exit /b %ROOP_STOP_EXIT%
