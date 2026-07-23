@echo off
title ETS2 + Telemetry Server Launcher
color 0A

echo.
echo  =====================================================
echo   ETS2 Virtual Steering Wheel — Game Launcher
echo  =====================================================
echo.

REM ── Check ETS2 exists ──────────────────────────────────────────────────────
set ETS2_EXE=G:\Euro Truck Simulator 2\bin\win_x64\eurotrucks2.exe
set TELEM_EXE=G:\Euro Truck Simulator 2\ets2-telemetry-server-master\server\Ets2Telemetry.exe

if not exist "%ETS2_EXE%" (
    echo  [ERROR] ETS2 not found at:
    echo          %ETS2_EXE%
    echo.
    echo  Edit this .bat file and update ETS2_EXE to the correct path.
    pause
    exit /b 1
)

if not exist "%TELEM_EXE%" (
    echo  [ERROR] Funbit telemetry server not found at:
    echo          %TELEM_EXE%
    echo.
    echo  Edit this .bat file and update TELEM_EXE to the correct path.
    pause
    exit /b 1
)

REM ── Start Funbit Telemetry Server ───────────────────────────────────────────
echo  [1/3] Starting Funbit Telemetry Server...
start "" "%TELEM_EXE%"
echo       Server started. Dashboard: http://192.168.56.1:25555
echo.

REM ── Wait a moment for the server to initialise ──────────────────────────────
timeout /t 3 /nobreak >nul

REM ── Start ETS2 ─────────────────────────────────────────────────────────────
echo  [2/3] Starting Euro Truck Simulator 2...
start "" "%ETS2_EXE%"
echo       ETS2 launched.
echo.

REM ── Wait for ETS2 to load before launching the wheel app ───────────────────
echo  [3/3] Waiting 10 seconds for ETS2 to load...
timeout /t 10 /nobreak >nul

REM ── Start the Virtual Steering Wheel app ────────────────────────────────────
echo  Starting Virtual Steering Wheel app...
echo.

REM  Try to find Python 3.11 in common locations
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.11 "%~dp0virtual_steering_wheel.py"
) else (
    echo  [ERROR] Python launcher (py) not found.
    echo  Install Python 3.11 from https://www.python.org and check "Add to PATH".
    pause
)
