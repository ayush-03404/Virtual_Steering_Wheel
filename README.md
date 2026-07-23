# 🎮 Virtual Steering Wheel for Euro Truck Simulator 2

Control **Euro Truck Simulator 2** using your phone as a steering wheel — tilt it to steer, tap buttons for gas, brake, gear shifts, indicators, and horn. Optionally add webcam hand/face tracking for a fully hands-free experience.

**Works with the ETS2 free demo and the full game.**

---

## 📋 Table of Contents

1. [What You Need](#1-what-you-need)
2. [Installation — Step by Step](#2-installation--step-by-step)
   - [Step A — Install ViGEm Bus Driver](#step-a--install-vigem-bus-driver)
   - [Step B — Install Python 3.11](#step-b--install-python-311)
   - [Step C — Install Python packages](#step-c--install-python-packages)
   - [Step D — Install ETS2 (demo or full)](#step-d--install-ets2-demo-or-full)
   - [Step E — Install Funbit Telemetry Server](#step-e--install-funbit-telemetry-server)
3. [First Launch](#3-first-launch)
4. [Connecting Your Phone](#4-connecting-your-phone)
5. [Playing — Phone Mode](#5-playing--phone-mode)
6. [Playing — Camera Mode (optional)](#6-playing--camera-mode-optional)
7. [Live Gauges on Your Phone](#7-live-gauges-on-your-phone)
8. [Troubleshooting](#8-troubleshooting)
9. [File Reference](#9-file-reference)

---

## 1. What You Need

| Item | Where to get it | Cost |
|---|---|---|
| **Windows 10/11 PC (64-bit)** | Your computer | — |
| **Euro Truck Simulator 2** | Steam (demo is free, full game ~$20) | Free demo |
| **Python 3.11 (64-bit)** | python.org | Free |
| **ViGEm Bus Driver** | GitHub (nefarius/ViGEmBus) | Free |
| **Funbit ETS2 Telemetry Server** | GitHub (Funbit/ets2-telemetry-server) | Free |
| **A smartphone** | Any iPhone or Android with a browser | — |
| **Both on the same Wi-Fi** | Your home router | — |
| *(Optional)* **Webcam** | Built-in laptop camera or USB | — |

---

## 2. Installation — Step by Step

### Step A — Install ViGEm Bus Driver

This free Windows driver lets Python create a virtual Xbox controller that ETS2 reads as a real gamepad.

1. Go to: **https://github.com/nefarius/ViGEmBus/releases**
2. Download the latest **`ViGEmBus_Setup_x64.exe`**
3. Run it and click through the installer (takes ~30 seconds)
4. **Restart your PC** after it finishes

> ⚠️ Without this driver, ETS2 will not receive any steering input from the app.

---

### Step B — Install Python 3.11

> ⚠️ Must be **Python 3.11 specifically** — not 3.12, not 3.10. Other versions have package compatibility issues.

1. Go to: **https://www.python.org/downloads/release/python-3110/**
2. Scroll down to **Files** → click **Windows installer (64-bit)**
3. Run the installer
4. On the first screen: ✅ **Tick "Add Python to PATH"** (critical — do this before clicking Install Now)
5. Click **Install Now**

Verify it worked — open Command Prompt and type:
```
py -3.11 --version
```
Should print `Python 3.11.x`. If you get an error, Python was not added to PATH — reinstall and tick the checkbox.

---

### Step C — Install Python Packages

Open **Command Prompt** and run this single command:

```
py -3.11 -m pip install mediapipe==0.10.14 opencv-python vgamepad pynput numpy websockets cryptography Pillow "qrcode[pil]"
```

This installs everything in one go. It will take 2–5 minutes on first run.

> 💡 If `mediapipe` fails, try: `py -3.11 -m pip install mediapipe==0.10.14 --no-deps` then install the rest separately.

---

### Step D — Install ETS2 (Demo or Full)

#### Free Demo

1. Install **Steam** from **https://store.steampowered.com/about/**
2. Create a free Steam account if you don't have one
3. Open Steam → click the search bar → search **Euro Truck Simulator 2**
4. On the store page, click **"Download Demo"** (below the Buy button)
5. Install and launch it once to confirm it works, then close it

#### Full Game

Buy it through Steam normally. It goes on sale for under $5 regularly. Everything works identically — the full game additionally supports DLC maps and longer routes.

---

### Step E — Install Funbit ETS2 Telemetry Server

This is the server that reads ETS2's live data (speed, RPM, gear) and makes it available to the phone dashboard. **It works with both the demo and the full game.**

1. Go to: **https://github.com/Funbit/ets2-telemetry-server/releases**
2. Download the latest **`ets2-telemetry-server-master.zip`** (or the zip from the releases page)
3. Extract it anywhere — e.g. inside your ETS2 game folder:
   ```
   G:\Euro Truck Simulator 2\ets2-telemetry-server-master\
   ```
4. Inside the extracted folder, go to `server\` — you should see `Ets2Telemetry.exe`
5. **Copy the plugin DLL into ETS2:**
   - Inside the extracted folder → go to `server\plugins\`
   - Copy **`scs-telemetry.dll`** (the Win64 build)
   - Paste it into:
     ```
     C:\Users\YOUR_NAME\Documents\Euro Truck Simulator 2\plugins\
     ```
   - Create the `plugins` folder if it doesn't exist
6. Launch ETS2 once after this — it will load the plugin automatically on startup

> ✅ You do **not** need to do any extra configuration. The server's default settings work out of the box.

---

## 3. First Launch

Everything is started with a single batch file:

1. Open the **`virtual_steering_wheel_app`** folder
2. Double-click **`start_ets2.bat`**

This will automatically:
- Start the **Funbit Telemetry Server** (small window opens)
- Launch **Euro Truck Simulator 2**
- Wait 10 seconds for the game to load
- Open the **Virtual Steering Wheel** app

> 💡 The first time you run it, Windows may show a SmartScreen warning for `start_ets2.bat`. Click **"More info"** → **"Run anyway"** — it's safe.

#### What you should see

| Window | Means |
|---|---|
| Black console window (Funbit server) | Telemetry server is running ✓ |
| ETS2 game window opens | Game is launching ✓ |
| Blue/dark Virtual Steering Wheel app | App is ready ✓ |

> ⚠️ **If ETS2 or the server are already running** from a previous session, close them first before running the bat again — it won't start a second copy, but it's cleaner to start fresh.

---

## 4. Connecting Your Phone

Your phone becomes the steering wheel. It connects over your local Wi-Fi.

**Requirements:**
- Phone and PC must be on the **same Wi-Fi network**
- Phone must have a browser (Chrome on Android, Safari on iPhone)

**Steps:**

1. In the Virtual Steering Wheel app, click **"Phone Mode"**
2. Click **"Start Server"**
3. A **QR code** appears on screen
4. On your phone: open the camera app and scan the QR code
   - OR open your browser and type the URL shown under the QR code (e.g. `https://192.168.x.x:8765`)
5. Your browser will show a security warning ("certificate not trusted") — this is expected for the local HTTPS connection
   - **Chrome/Android:** tap **"Advanced"** → **"Proceed to ... (unsafe)"**
   - **Safari/iPhone:** tap **"Show Details"** → **"visit this website"** → confirm
6. The phone UI loads — you should see **"Connected ✓"** at the top

> ⚠️ If the QR code scan doesn't open the page, type the URL manually in your phone's browser.

---

## 5. Playing — Phone Mode

Hold your phone **horizontally (landscape)** with the top edge pointing to the right.

### Controls

| Action | How to do it |
|---|---|
| **Steer left** | Tilt phone down (top goes lower) |
| **Steer right** | Tilt phone up (top goes higher) |
| **Gas** | Tap and hold the **GAS** button (bottom right) |
| **Brake** | Tap and hold the **BRAKE** button (bottom left) |
| **Shift up** | Tap **▲** (instant — one tap per shift) |
| **Shift down** | Tap **▼** (instant — one tap per shift) |
| **Horn** | Tap **HORN** |
| **Left indicator** | Tap **◄** |
| **Right indicator** | Tap **►** |
| **Lights** | Tap **LIGHTS** |

### Sensitivity Profiles

Tap **P1 / P2 / P3** at the top of the phone UI to switch steering sensitivity:
- **P1** — gentle (good for highways)
- **P2** — normal
- **P3** — responsive (good for tight turns)

### Tips for best control

- Keep the phone flat (not tilted toward you/away from you) — only left/right tilt steers
- Rest your elbows on a desk or your knees so your arms don't get tired
- Use **Manual transmission** in ETS2 settings for the most satisfying experience with the shift buttons
- Set ETS2 controller type to **Gamepad / Xbox Controller** in Options → Controls

### Setting up ETS2 to use the virtual controller

1. Open ETS2 → **Options** → **Controls**
2. Set **Controller** to: `Xbox 360 Controller` (or `Gamepad`)
3. Left stick → steering should map automatically
4. If it doesn't steer: click on the Steering axis and move your phone left/right to assign it

---

## 6. Playing — Camera Mode (optional)

Camera mode uses your webcam and MediaPipe AI to track your hands and face. Your hands become the steering wheel and your mouth controls the horn.

**Requirements:** A webcam (built-in or USB)

1. In the app, click **"Camera Mode"**
2. Click **"Start Camera"**
3. A preview window opens showing your camera
4. Hold both hands up in front of the camera

### Camera controls

| Action | How |
|---|---|
| **Steer** | Tilt your hands left or right (like holding a real wheel) |
| **Gas** | Open your right hand flat |
| **Brake** | Close your right hand to a fist |
| **Horn** | Open your mouth wide |

### Camera tips

- Make sure you're well-lit — face and hands must both be clearly visible
- Keep your hands roughly at chest height, not too close or far from the camera
- A plain background helps tracking accuracy

---

## 7. Live Gauges on Your Phone

When everything is set up correctly, the phone UI shows live **speed**, **RPM**, and **gear** pulled directly from ETS2.

The top-right corner of the phone UI shows the connection status:

| Indicator | Meaning |
|---|---|
| **● LIVE** (green) | Live data from ETS2 — gauges are real |
| **● SIM (demo)** (grey) | Telemetry server not connected — gauges are simulated |

### If gauges show SIM instead of LIVE

Check in this order:

1. **Is `start_ets2.bat` running?** — the Funbit server window must be open
2. **Is ETS2 fully loaded into a save?** — the server only reads data when you're actually driving, not on the main menu
3. **Did you copy `scs-telemetry.dll` into the plugins folder?** — see Step E above
4. **Did ETS2 load after the DLL was placed?** — the plugin loads on game startup; you need to restart ETS2 after first placing the DLL

Open a browser on your PC and go to:
```
http://192.168.56.1:25555/api/ets2/telemetry
```
If you see JSON data with `"connected": true`, the server is working. If it's `"connected": false`, ETS2 isn't in a driving session yet (still on the menu). If the page doesn't load at all, the Funbit server isn't running.

---

## 8. Troubleshooting

### "Connected ✓" never appears on the phone

- Make sure phone and PC are on **the same Wi-Fi** (not one on 5 GHz and one on 2.4 GHz — try switching them to match)
- Some routers block devices from talking to each other — try temporarily disabling the router firewall, or connect both to a mobile hotspot
- Make sure you accepted the browser security warning on the phone

### Phone shows "Reconnecting…" constantly

- The app server on the PC stopped — check the app is still open and running
- Restart the app and scan the QR code again

### ETS2 doesn't respond to steering

- ViGEm Bus Driver is not installed — install it from the link in Step A and restart your PC
- Go to ETS2 → Options → Controls → reassign the steering axis

### Gauges are always SIM / not going LIVE

- Funbit server is not running — check the black server window is open
- The plugin DLL is missing — re-read Step E and confirm `scs-telemetry.dll` is in the plugins folder
- ETS2 is on the main menu — load a save/game and start driving first
- Verify: open `http://192.168.56.1:25555` in a browser — if it doesn't open, the server isn't running

### "vgamepad not installed" error on startup

```
py -3.11 -m pip install vgamepad
```
Then confirm ViGEm Bus Driver is installed (Step A).

### Camera mode crash / MediaPipe error

```
py -3.11 -m pip install mediapipe==0.10.14 opencv-python --upgrade
```
Make sure you're using Python **3.11 specifically** — MediaPipe 0.10.14 does not support Python 3.12+.

### "Port already in use" error

Another instance of the app is already running. Close all Python windows and relaunch.

### start_ets2.bat says ETS2 not found

Edit `start_ets2.bat` in Notepad and update the `ETS2_EXE` path on line 9 to match where your game is installed.

---

## 9. File Reference

| File | Purpose |
|---|---|
| `virtual_steering_wheel.py` | **Main app** — camera + phone modes in one window |
| `start_ets2.bat` | **One-click launcher** — starts ETS2, Funbit server, and the app |
| `steering_wheel.py` | Camera-only standalone (no GUI) |
| `phone_server.py` | Phone-only standalone (no GUI) |
| `build.bat` | Builds a standalone `.exe` with PyInstaller |
| `README.md` | This guide |

---

## Quick-Start Checklist

- [ ] ViGEm Bus Driver installed + PC restarted
- [ ] Python 3.11 (64-bit) installed with "Add to PATH" ticked
- [ ] All Python packages installed (`pip install ...` command from Step C)
- [ ] ETS2 (demo or full) installed via Steam
- [ ] Funbit telemetry server extracted and DLL copied to ETS2 plugins folder
- [ ] ETS2 launched at least once after placing the DLL (to load the plugin)
- [ ] `start_ets2.bat` used to launch everything
- [ ] Phone and PC on the same Wi-Fi
- [ ] ETS2 Controls set to Gamepad/Xbox Controller

Once all boxes are checked, you're ready to drive. 🚛
