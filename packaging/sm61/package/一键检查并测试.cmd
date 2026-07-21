@echo off
setlocal
chcp 65001 >nul
title GTX 1080 Ti ORT TensorRT Runtime Test

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-and-test.ps1"
set "_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%_EXIT_CODE%"=="0" (
  echo Test completed successfully.
) else (
  echo Test failed. Check the logs directory and README.
)
echo Press any key to close this window.
pause >nul
exit /b %_EXIT_CODE%
