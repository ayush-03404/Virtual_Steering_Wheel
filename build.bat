@echo off
echo ============================================================
echo  Virtual Steering Wheel — Install + Build
echo ============================================================
echo.

echo [1/2] Installing dependencies...
py -3.11 -m pip install --upgrade pip --quiet
py -3.11 -m pip install --quiet ^
    mediapipe==0.10.14 ^
    opencv-python ^
    pynput ^
    vgamepad ^
    numpy ^
    pillow ^
    matplotlib ^
    pyqt5 ^
    pyinstaller

echo.
echo [2/2] Building EXE...
set QT_API=pyqt5
py -3.11 -m PyInstaller ^
    --onedir ^
    --windowed ^
    --noconfirm ^
    --noupx ^
    --hidden-import=vgamepad ^
    --hidden-import=PIL ^
    --hidden-import=PIL._imagingtk ^
    --hidden-import=matplotlib ^
    --collect-all=vgamepad ^
    --collect-all=mediapipe ^
    --collect-all=matplotlib ^
    --collect-all=PIL ^
    --collect-data=mediapipe ^
    --collect-binaries=mediapipe ^
    steering_wheel.py

echo.
echo ============================================================
echo  Done!
echo  Your app folder is:  dist\steering_wheel\
echo  Run it with:         dist\steering_wheel\steering_wheel.exe
echo.
echo  IMPORTANT: Also install ViGEm Bus Driver if not yet done:
echo  https://github.com/nefarius/ViGEmBus/releases
echo ============================================================
pause
