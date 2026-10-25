@echo off
chcp 65001 >nul
title Leaf API Server - 4 models
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

echo ==================================================
echo   Leaf API  -  one-click launcher
echo   Models : leaf1.0 / leaf-things1.0 / leaf-flash / leaf-code
echo   Auth   : keys listed in api.json - default: leaf-sk-123456
echo   Stop   : close this window or press Ctrl+C
echo ==================================================
echo.

where python >nul 2>nul
if %errorlevel%==0 (
    python api_server.py %*
    goto :done
)
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 api_server.py %*
    goto :done
)
echo [ERROR] Python not found. Install Python 3 and check "Add to PATH".

:done
echo.
echo Server exited.
pause
