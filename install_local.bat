@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_local.ps1" %*
set "ROOP_INSTALL_EXIT=%ERRORLEVEL%"
if not "%ROOP_INSTALL_EXIT%"=="0" pause
exit /b %ROOP_INSTALL_EXIT%
