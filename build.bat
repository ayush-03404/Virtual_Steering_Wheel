@echo off
title Virtual Steering Wheel — Build EXE
color 0A

echo.
echo  =====================================================
echo   Virtual Steering Wheel -- Install and Build EXE
echo  =====================================================
echo.
echo  NOTE: This builds the Phone Mode EXE (tilt-to-steer).
echo  Camera mode requires running from Python directly.
echo.

REM ── Install only what the Phone-mode EXE needs ─────────────────────────────
echo  [1/2] Installing dependencies...
echo.
py -3.11 -m pip install --upgrade pip --quiet
py -3.11 -m pip install --quiet ^
    vgamepad ^
    pynput ^
    numpy ^
    websockets ^
    cryptography ^
    Pillow ^
    "qrcode[pil]" ^
    pyinstaller

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] pip install failed. Check your internet and try again.
    pause
    exit /b 1
)

echo.
echo  [2/2] Building EXE...
echo  (should finish in 1-3 minutes)
echo.

py -3.11 -m PyInstaller ^
    --onedir ^
    --windowed ^
    --noconfirm ^
    --noupx ^
    --name "VirtualSteeringWheel" ^
    --hidden-import=vgamepad ^
    --hidden-import=pynput.keyboard ^
    --hidden-import=pynput.mouse ^
    --hidden-import=PIL._imagingtk ^
    --hidden-import=PIL.Image ^
    --hidden-import=qrcode ^
    --hidden-import=websockets ^
    --hidden-import=cryptography ^
    --hidden-import=urllib.request ^
    --collect-all=vgamepad ^
    --collect-all=pynput ^
    --exclude-module=mediapipe ^
    --exclude-module=cv2 ^
    --exclude-module=matplotlib ^
    --exclude-module=PyQt5 ^
    --exclude-module=PyQt6 ^
    --exclude-module=PySide2 ^
    --exclude-module=PySide6 ^
    --exclude-module=scipy ^
    --exclude-module=pandas ^
    --exclude-module=tensorflow ^
    --exclude-module=torch ^
    --exclude-module=torchvision ^
    --exclude-module=sklearn ^
    --exclude-module=IPython ^
    virtual_steering_wheel.py

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Build failed. See errors above.
    pause
    exit /b 1
)

echo.
echo  =====================================================
echo   Build complete!
echo.
echo   Run:  dist\VirtualSteeringWheel\VirtualSteeringWheel.exe
echo.
echo   IMPORTANT: ViGEm Bus Driver must be installed:
echo   https://github.com/nefarius/ViGEmBus/releases
echo  =====================================================
echo.
pause
