# 🎮 Virtual Steering Wheel — Full Setup Guide

Control **Euro Truck Simulator 2** (or any driving game) using your hands and face as a controller. No physical steering wheel needed — just a webcam and your hands.

---

## 📋 What You Need (Before Starting)

| Requirement | Link | Notes |
|---|---|---|
| **Steam** | https://store.steampowered.com/about/ | Free |
| **Euro Truck Simulator 2** | https://store.steampowered.com/app/227300/Euro_Truck_Simulator_2/ | Paid game — **free demo available** (see below) |
| **Python 3.11** | https://www.python.org/downloads/release/python-3110/ | **Must be 3.11** — not 3.12, not 3.10 |
| **ViGEm Bus Driver** | https://github.com/nefarius/ViGEmBus/releases | Free Windows driver — install this first |
| **A webcam** | Built-in laptop camera or USB webcam | Positioned so your hands and face are both visible |

---

## 🚀 Step-by-Step Setup

### Step 1 — Install Steam and ETS2

> 💡 **Just trying it out? Use the free demo first.**
> ETS2 has an official free demo on Steam — no purchase needed.
> Download it here: https://store.steampowered.com/app/227300/Euro_Truck_Simulator_2/
> On the store page click **"Download Demo"** (below the Buy button).
> The demo includes a short route and is fully compatible with this steering wheel project.
> If you enjoy it, buy the full game — it goes on sale for under $5 regularly.

1. Download and install Steam from https://store.steampowered.com/about/
2. Create a free Steam account if you don't have one
3. Install **Euro Truck Simulator 2** — full game **or** free demo (see above)
4. Launch it once to make sure it runs, then close it

---

### Step 2 — Install ViGEm Bus Driver

This is a free Windows system driver that lets Python create a virtual Xbox controller. **ETS2 reads your steering through this.**

1. Go to https://github.com/nefarius/ViGEmBus/releases
2. Download the latest `ViGEmBus_Setup_x64.exe`
3. Run it and click through the installer
4. **Restart your PC** after installation

---

### Step 3 — Install Python 3.11

1. Go to https://www.python.org/downloads/release/python-3110/
2. Scroll down to **Files** → download `Windows installer (64-bit)`
3. Run the installer
4. ✅ **Important:** On the first screen, check **"Add Python to PATH"** before clicking Install

To verify:
```
py -3.11 --version
```
Should print: `Python 3.11.x`

---

### Step 4 — Download the Project

1. Download the `virtual_steering_wheel.zip`
2. Right-click the zip → **Extract All**
3. Extract it somewhere simple, e.g. `C:\SteeringWheel\`

Your folder should contain:
```
steering_wheel.py
requirements.txt
README.md
build.bat
```

---

### Step 5 — Run build.bat

1. **Double-click `build.bat`**
2. A black console window opens — it installs all Python packages, then compiles the EXE
3. This takes **3–10 minutes** the first time (MediaPipe is large)
4. When done you'll see: `Done! Your app folder is: dist\steering_wheel\`

Your EXE is at:
```
dist\steering_wheel\steering_wheel.exe
```

> 💡 You can also just run `steering_wheel.py` directly with Python at any time — no build needed.

---

### Step 6 — Configure ETS2 Controls

This is the most important step. ETS2 needs to know to read your virtual Xbox controller for steering.

1. Launch ETS2 via Steam
2. Go to **Options → Controls**
3. At the top, change **Control method** to: **`Xbox Controller`** (or `Gamepad`)
4. Scroll to **Steering** → click the axis field → move your hands (with the script running) to assign it
5. Set the **Steering sensitivity** slider to around **60–70%**
6. Set **Non-linearity** to **0%** (the script handles the curve itself)
7. Click **Accept**

**Brake/Gas** use keyboard arrows, so those work automatically — no binding needed.

---

### Step 7 — Launch and Drive

1. Open ETS2 and start a drive (leave the game running in the background)
2. **Double-click** `dist\steering_wheel\steering_wheel.exe`
3. A camera window opens — position yourself so **both hands AND your face** are visible
4. Hold your hands like you're gripping a real steering wheel
5. The HUD shows your steering bar, gas/brake, and FPS

**To quit:** Press **Q** or **Escape** in the camera window.

---

## 🙌 Gesture Reference

| Gesture | Action |
|---|---|
| **Tilt both hands left/right** | Steering (analog — proportional to tilt angle) |
| **Mouth closed** | Gas (Up arrow held) |
| **Mouth open** | Brake (Down arrow held) |
| **Raise both eyebrows (hold)** | Horn (H key held) |
| **Tilt head right** | Left indicator toggle |
| **Tilt head left** | Right indicator toggle |

> 💡 **Steering tip:** Hold your hands out in front of you like a real steering wheel. The camera should see both wrists from slightly above. Even a small tilt registers — you don't need to spin your hands dramatically.

---

## 🎛️ Speed Profiles

Press **1**, **2**, or **3** in the camera window (click it first to give it focus):

| Key | Profile | Full Lock At | Best For |
|---|---|---|---|
| `1` | CITY | 28° tilt | City streets, tight turns |
| `2` | HIGHWAY | 26° tilt | Open roads |
| `3` | MOTORWAY | 16° tilt | Highways, keeping a trailer stable |

> Start with **profile 3** when driving with a trailer — it makes small steering corrections much easier.

---

## 📷 Camera Position Tips

- Sit **0.5–1 metre** from the camera
- Camera should be at **chest height or slightly below**, angled slightly up
- Make sure **both hands AND your face** are in frame at the same time
- **Good lighting matters** — avoid backlighting (window behind you). A lamp in front of you works best
- If using a **USB webcam**, plug it in before launching the script

---

## ❌ Troubleshooting

### Camera / Video Problems

| Problem | Fix |
|---|---|
| **Black screen / no camera window** | Another app (Teams, OBS, Discord) is using the camera. Close them and try again |
| **Wrong camera opens** (shows external cam when you want built-in) | The script auto-detects — unplug the USB camera to force it to use the built-in one, or replug to switch |
| **Camera opens but no image** | Update your camera drivers in Device Manager |
| **Very low FPS / laggy** | Close background apps; make sure no other app is using the camera |
| **Hands not detected** | Improve lighting; make sure both hands are clearly visible; move closer to the camera |
| **Face not detected** | Ensure your face is in frame; don't wear a mask or hat that covers your brow |

---

### Installation / Build Problems

| Problem | Fix |
|---|---|
| **`py -3.11` not recognized** | Python 3.11 not installed or not added to PATH — reinstall and check "Add to PATH" |
| **`build.bat` closes instantly** | Right-click it → **Run as Administrator** |
| **PyInstaller fails with mediapipe error** | Run `py -3.11 -m pip install mediapipe==0.10.14` manually first, then re-run build.bat |
| **EXE won't start — missing DLL error** | Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| **Antivirus deletes the EXE** | Add the `dist\steering_wheel\` folder to your antivirus exclusions (PyInstaller EXEs often trigger false positives) |

---

### Game / Controller Problems

| Problem | Fix |
|---|---|
| **Steering does nothing in ETS2** | Make sure ViGEm Bus Driver is installed and you've rebooted; check ETS2 Controls → set method to Xbox Controller |
| **Steering is too sensitive** | Press **3** (Motorway profile) in the camera window |
| **Steering is not sensitive enough** | Press **1** (City profile) in the camera window |
| **Trailer keeps falling / oversteering** | Use profile **3**; keep your hands steady at centre more often |
| **Horn keeps firing by itself** | Avoid raising your eyebrows naturally — or increase `BROW_RAISE_THRESH` in the script (default 0.65, try 0.75) |
| **Indicators trigger when not intended** | Increase `HEAD_TILT_THRESH` in the script (default 0.05, try 0.07) |
| **Gas/brake stuck** | The script releases all keys on quit (Q). If stuck, alt-tab to ETS2 and press the arrow keys once to reset |

---

### Internal vs External Webcam

The script **auto-detects** your camera — it scans indices 0 through 4 and picks the first one that works.

- If it opens the **wrong camera**, simply **unplug the other one** and restart the script
- If you have both plugged in and want to choose, open `steering_wheel.py` in Notepad and change:
  ```python
  CAMERA_INDEX = None   # None = auto-detect
  ```
  to a specific number:
  ```python
  CAMERA_INDEX = 1      # 0 = usually built-in, 1 = usually first USB camera
  ```

---

## ⚙️ Advanced Config

Open `steering_wheel.py` in Notepad to tweak these values at the top:

| Setting | Default | What It Does |
|---|---|---|
| `CAMERA_INDEX` | `None` | `None` = auto-detect; set to `0`, `1`, etc. to force a specific camera |
| `DEAD_ZONE_DEG` | `4` | Degrees of tilt ignored at centre — increase if steering drifts |
| `STEERING_CURVE` | `3.0` | `1` = linear, `3` = cubic (gentler at small angles) |
| `EMA_ALPHA` | `0.20` | Smoothing: lower = smoother but more lag; higher = more responsive |
| `BROW_RAISE_THRESH` | `0.65` | How extreme a brow raise is needed for horn — raise if accidental |
| `HEAD_TILT_THRESH` | `0.05` | How much head tilt triggers indicator — raise if accidental |
| `FLIP_CAMERA` | `True` | Mirror the feed (selfie mode). Try `False` if steering feels reversed |

---

## 🎮 Works With Other Games Too

Any game that supports an **Xbox controller** or **keyboard arrows** will work:

- **Keyboard arrows**: Mouth open/closed controls gas/brake automatically
- **Xbox controller axis**: Steering uses analog left-stick X — works in any game that supports a controller

Games tested:
- Euro Truck Simulator 2 ✅
- American Truck Simulator ✅
- Any browser racing game using arrow keys ✅
