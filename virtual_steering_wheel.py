"""
Virtual Steering Wheel — All-in-One App
========================================
Camera mode  : webcam + MediaPipe hand/face tracking → steering / gas / horn / indicators
Phone mode   : hold phone in landscape, tilt to steer; on-screen buttons for everything

Steering direction is identical in both modes:
  tilt/tilt-hands RIGHT  →  turn RIGHT in ETS2
  tilt/tilt-hands LEFT   →  turn LEFT  in ETS2

Live ETS2 gauges via Funbit ETS2 Telemetry Server (works with demo + full game).
  Start     : run start_ets2.bat  (launches ETS2 + Funbit server together)
  API URL   : http://192.168.56.1:25555/api/ets2/telemetry

Requirements
  pip install mediapipe==0.10.14 opencv-python vgamepad pynput numpy
              websockets cryptography Pillow qrcode[pil]
  Windows  : ViGEm Bus Driver  https://github.com/nefarius/ViGEmBus/releases

Run
  py -3.11 virtual_steering_wheel.py
"""

from __future__ import annotations
import asyncio, json, math, os, platform, queue, random
import socket, ssl, subprocess, sys, threading, time
import urllib.request, urllib.error
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from tkinter import scrolledtext

# ── Optional: Camera / MediaPipe ───────────────────────────────────────────────
try:
    import cv2, numpy as np
    from mediapipe.python.solutions import (
        hands        as _mp_hands,
        face_mesh    as _mp_face,
        drawing_utils as _mp_draw,
    )
    _CAM_LIBS = True
except ImportError:
    _CAM_LIBS = False

# ── Optional: PIL (camera frame display + QR image) ───────────────────────────
try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

# ── Optional: QR code ─────────────────────────────────────────────────────────
try:
    import qrcode as _qr_mod
    _QRCODE = True
except ImportError:
    _QRCODE = False

# ── Optional: Virtual gamepad ─────────────────────────────────────────────────
try:
    import vgamepad as vg
    _GAMEPAD = True
except ImportError:
    vg = None
    _GAMEPAD = False

# ── Optional: Keyboard ────────────────────────────────────────────────────────
try:
    from pynput.keyboard import Key as _Key, Controller as _KbCtrl
    _KB = True
except ImportError:
    _KB = False

# ── Optional: WebSocket ───────────────────────────────────────────────────────
try:
    import websockets
    _WS = True
except ImportError:
    _WS = False


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

HTTP_PORT  = 8443
WS_PORT    = 8765
CERT_FILE  = "phone_cert.pem"
KEY_FILE   = "phone_key.pem"

CAMERA_INDEX       = None   # None = auto-detect; set to 0/1/2 to force a specific camera
FLIP_CAMERA        = True   # mirror image (selfie view)
MIN_DETECTION_CONF = 0.3
MIN_TRACKING_CONF  = 0.2
PROCESS_SCALE      = 0.50   # downscale factor for MediaPipe inference (speed vs accuracy)
GRACE_FRAMES       = 30     # frames before steering eases to centre after hands leave frame
SHOW_ANGLE         = True

DEAD_ZONE_DEG  = 5     # degrees of tilt ignored at centre — increase if steering drifts
STEERING_CURVE = 2.5   # 1=linear (harsh), 2.5=gentler at small angles (recommended)
EMA_ALPHA      = 0.20  # smoothing: lower=smoother/laggier, higher=more responsive
RECENTER_RATE  = 0.05  # axis units per frame when hands leave frame (~1 s full→centre)

# Speed profiles — soft_zone = tilt angle (°) that produces full-lock steering
# Camera uses ord() keys; Phone uses integer keys.  Values kept in sync below.
_CAM_PROFILES = {
    ord('1'): dict(name='CITY',     soft_zone=70),
    ord('2'): dict(name='HIGHWAY',  soft_zone=55),
    ord('3'): dict(name='MOTORWAY', soft_zone=40),
}
_PHONE_PROFILES = {
    1: dict(name='CITY',     soft_zone=70),
    2: dict(name='HIGHWAY',  soft_zone=55),
    3: dict(name='MOTORWAY', soft_zone=40),
}

MOUTH_OPEN_THRESH   = 0.38
MOUTH_CLOSE_THRESH  = 0.28
BROW_RAISE_THRESH   = 0.65
BROW_CONFIRM_FRAMES = 8
HEAD_TILT_THRESH    = 0.05
HEAD_TILT_RETURN    = 0.02
HEAD_TILT_CONFIRM   = 6
HEAD_TILT_COOLDOWN  = 1.5

KEY_HORN       = 'h'
KEY_IND_LEFT   = ','
KEY_IND_RIGHT  = '.'
KEY_LIGHTS     = 'l'
KEY_ENGINE     = 'e'         # ETS2: start/stop engine
KEY_CRUISE     = 'c'         # ETS2: cruise control toggle
KEY_HAZARD     = 'f'         # ETS2: hazard lights — assign F key in ETS2 Controls
KEY_PARK_BRAKE = ' '         # ETS2: parking brake (spacebar)
KEY_DIFF_LOCK  = 'v'         # ETS2: differential lock
KEY_CAM_VIEWS  = ['1', '2', '3']  # ETS2 camera view cycle (interior → outside → top)


# ════════════════════════════════════════════════════════════════════════════════
# STEERING MATH  (shared between camera mode and phone mode)
# ════════════════════════════════════════════════════════════════════════════════

def tilt_to_axis(angle_deg: float, soft_zone: float) -> float:
    """
    Maps a tilt angle (degrees) to a joystick axis value in [-1, +1].

    Convention used consistently in both modes:
      positive angle → positive axis → RIGHT turn in ETS2
      negative angle → negative axis → LEFT  turn in ETS2

    The phone mode negates the raw DeviceOrientation value so that
    'tilt phone right' produces the same positive result as 'tilt hands right'.
    """
    if abs(angle_deg) < DEAD_ZONE_DEG:
        return 0.0
    sign = 1.0 if angle_deg > 0 else -1.0
    raw  = (abs(angle_deg) - DEAD_ZONE_DEG) / max(soft_zone - DEAD_ZONE_DEG, 1)
    return sign * min(raw, 1.0) ** STEERING_CURVE


# ════════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS  (camera mode — MediaPipe landmark math)
# ════════════════════════════════════════════════════════════════════════════════

_MOUTH_TOP, _MOUTH_BOTTOM = 13, 14
_MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
_L_EYE_OUTER, _L_EYE_INNER = 263, 362
_R_EYE_OUTER, _R_EYE_INNER = 133,  33
_L_BROW_INNER, _R_BROW_INNER = 105, 334
_NOSE_TIP, _CHIN = 1, 152


def _head_tilt(lms) -> float:
    """Positive = head tilted right, negative = head tilted left."""
    ly = (lms[_L_EYE_OUTER].y + lms[_L_EYE_INNER].y) / 2.0
    ry = (lms[_R_EYE_OUTER].y + lms[_R_EYE_INNER].y) / 2.0
    return ry - ly


def _mouth_ar(lms, w: int, h: int) -> float:
    top    = np.array([lms[_MOUTH_TOP   ].x * w, lms[_MOUTH_TOP   ].y * h])
    bottom = np.array([lms[_MOUTH_BOTTOM].x * w, lms[_MOUTH_BOTTOM].y * h])
    left   = np.array([lms[_MOUTH_LEFT  ].x * w, lms[_MOUTH_LEFT  ].y * h])
    right  = np.array([lms[_MOUTH_RIGHT ].x * w, lms[_MOUTH_RIGHT ].y * h])
    return float(np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6))


def _brow_raise(lms) -> float:
    nose_y = lms[_NOSE_TIP].y; chin_y = lms[_CHIN].y
    brow_y = (lms[_L_BROW_INNER].y + lms[_R_BROW_INNER].y) / 2.0
    return (nose_y - brow_y) / max(abs(chin_y - nose_y), 1e-4)


# ════════════════════════════════════════════════════════════════════════════════
# CAMERA CONTROLLER
# ════════════════════════════════════════════════════════════════════════════════

class _CamCtrl:
    """Translates MediaPipe landmarks → gamepad/keyboard inputs for camera mode."""

    def __init__(self):
        if not _GAMEPAD:
            raise RuntimeError(
                'vgamepad not available.\n'
                '1. Install ViGEm Bus Driver: https://github.com/nefarius/ViGEmBus/releases\n'
                '2. pip install vgamepad')
        self.pad = vg.VX360Gamepad()
        self.kb  = _KbCtrl()

        self._steer     = 0.0
        self._ema_angle = 0.0
        self._mouth_open = False
        self._key_up = False; self._key_dn = False
        self._horn_held = False
        self._ind_left  = False; self._ind_right = False
        self._brow_frames = 0
        self._lwf = 0; self._rwf = 0
        self._lwt = 0.0; self._rwt = 0.0
        self._lwfire = False; self._rwfire = False

    # ── gamepad commit ────────────────────────────────────────────────────────
    def _commit(self):
        try:
            # Camera mode: positive steer = right turn (atan2 geometry is correct as-is)
            self.pad.left_joystick_float(x_value_float=self._steer, y_value_float=0.0)
            self.pad.update()
        except Exception:
            pass

    def _kp(self, key):
        if key == _Key.up   and not self._key_up: self.kb.press(_Key.up);   self._key_up = True
        if key == _Key.down and not self._key_dn: self.kb.press(_Key.down); self._key_dn = True

    def _kr(self, key):
        if key == _Key.up   and self._key_up: self.kb.release(_Key.up);   self._key_up = False
        if key == _Key.down and self._key_dn: self.kb.release(_Key.down); self._key_dn = False

    def _tap(self, c): self.kb.press(c); self.kb.release(c)

    # ── steering ──────────────────────────────────────────────────────────────
    def update_steering(self, lw: tuple, rw: tuple, soft_zone: float):
        """lw/rw = (norm_x, norm_y) of left/right wrist landmarks."""
        dx  = rw[0] - lw[0]
        dy  = rw[1] - lw[1]
        raw = math.degrees(math.atan2(dy, dx))
        self._ema_angle = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema_angle
        self._steer     = tilt_to_axis(self._ema_angle, soft_zone)
        self._commit()
        direction = 'LEFT' if self._steer < -0.05 else ('RIGHT' if self._steer > 0.05 else 'STRAIGHT')
        return self._ema_angle, direction

    def ease_to_center(self):
        if abs(self._steer) < RECENTER_RATE: self._steer = 0.0
        else: self._steer -= math.copysign(RECENTER_RATE, self._steer)
        self._commit()

    def release_steering(self):
        self._ema_angle = 0.0; self._steer = 0.0; self._commit()

    # ── throttle ──────────────────────────────────────────────────────────────
    def update_throttle(self, mar: float):
        if   mar > MOUTH_OPEN_THRESH:  self._mouth_open = True
        elif mar < MOUTH_CLOSE_THRESH: self._mouth_open = False
        if self._mouth_open: self._kp(_Key.down); self._kr(_Key.up)
        else:                self._kp(_Key.up);   self._kr(_Key.down)

    def release_throttle(self):
        self._kr(_Key.up); self._kr(_Key.down)

    # ── horn ──────────────────────────────────────────────────────────────────
    def update_horn(self, brow: float):
        if brow > BROW_RAISE_THRESH: self._brow_frames += 1
        else: self._brow_frames = 0
        should = self._brow_frames >= BROW_CONFIRM_FRAMES
        if should and not self._horn_held:
            self.kb.press(KEY_HORN);   self._horn_held = True
        elif not should and self._horn_held:
            self.kb.release(KEY_HORN); self._horn_held = False

    # ── indicators ────────────────────────────────────────────────────────────
    def update_indicators(self, tilt: float, now: float):
        # left tilt → right indicator
        if tilt < -HEAD_TILT_THRESH: self._lwf += 1
        else:
            if tilt > -HEAD_TILT_RETURN: self._lwf = 0; self._lwfire = False
        if self._lwf >= HEAD_TILT_CONFIRM and not self._lwfire and now - self._lwt > HEAD_TILT_COOLDOWN:
            self._ind_right = not self._ind_right
            self._lwfire = True; self._lwt = now; self._tap(KEY_IND_RIGHT)

        # right tilt → left indicator
        if tilt > HEAD_TILT_THRESH: self._rwf += 1
        else:
            if tilt < HEAD_TILT_RETURN: self._rwf = 0; self._rwfire = False
        if self._rwf >= HEAD_TILT_CONFIRM and not self._rwfire and now - self._rwt > HEAD_TILT_COOLDOWN:
            self._ind_left = not self._ind_left
            self._rwfire = True; self._rwt = now; self._tap(KEY_IND_LEFT)

    # ── release all ───────────────────────────────────────────────────────────
    def release_all(self):
        self.release_steering(); self.release_throttle()
        if self._horn_held: self.kb.release(KEY_HORN); self._horn_held = False

    @property
    def steer_axis(self) -> float: return self._steer
    @property
    def mouth_open(self) -> bool:  return self._mouth_open
    @property
    def horn_held(self) -> bool:   return self._horn_held
    @property
    def ind_left(self) -> bool:    return self._ind_left
    @property
    def ind_right(self) -> bool:   return self._ind_right


# ════════════════════════════════════════════════════════════════════════════════
# CAMERA STREAM  (background-thread capture for freshest frame)
# ════════════════════════════════════════════════════════════════════════════════

class _CamStream:
    def __init__(self, src, backend):
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None; self._lock = threading.Lock()
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self._t.start(); return self

    def _reader(self):
        while not self._stop.is_set():
            ret, f = self.cap.read()
            if ret and f is not None:
                with self._lock: self._frame = f

    def get(self):
        with self._lock:
            return (True, self._frame) if self._frame is not None else (False, None)

    def stop(self):
        self._stop.set(); self._t.join(timeout=2); self.cap.release()


def _find_camera(backend, preferred=None):
    candidates = list(range(5))
    if preferred is not None:
        candidates = [preferred] + [i for i in candidates if i != preferred]
    for idx in candidates:
        print(f'[CAM] Trying index {idx} ...', end=' ', flush=True)
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened(): print('not found'); cap.release(); continue
        for _ in range(3): cap.read()
        ok, f = cap.read()
        if ok and f is not None:
            h, w = f.shape[:2]; print(f'OK  ({w}×{h})'); return idx, cap
        print('no frame'); cap.release()
    raise RuntimeError(
        'No working camera found on indices 0–4.\n'
        '  • Make sure no other app (Teams, OBS, Discord) is using the camera.\n'
        '  • Unplug / replug USB webcam and retry.\n'
        '  Switch to Phone Mode if you have no webcam.')


# ════════════════════════════════════════════════════════════════════════════════
# CAMERA WORKER  (background thread — processes frames, draws HUD, queues output)
# ════════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Runs the camera+MediaPipe loop in a daemon thread.
    Pushes BGR numpy arrays (with HUD overlay drawn on them) into `frame_queue`
    for the GUI to display.  Updates `state['steer']` each frame.
    """

    def __init__(self, frame_queue: queue.Queue, state: dict):
        self._fq    = frame_queue
        self._state = state
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    def start(self): self._stop.clear(); self._t.start()
    def stop(self):  self._stop.set()

    # ── HUD drawing ───────────────────────────────────────────────────────────
    @staticmethod
    def _draw_wheel(frame, cx, cy, steer, angle):
        h, w   = frame.shape[:2]
        radius = int(min(w, h) * 0.10)
        clr    = (60,120,255) if steer < -0.05 else ((50,220,140) if steer > 0.05 else (200,200,200))
        cv2.circle(frame, (cx+3, cy+3), radius, (0,0,0), 4)
        cv2.circle(frame, (cx,   cy),   radius, clr,     3)
        for sa in (0, 120, 240):
            rad = math.radians(sa - angle)
            x1  = int(cx + radius * 0.40 * math.cos(rad))
            y1  = int(cy - radius * 0.40 * math.sin(rad))
            x2  = int(cx + radius * 0.95 * math.cos(rad))
            y2  = int(cy - radius * 0.95 * math.sin(rad))
            cv2.line(frame, (x1,y1), (x2,y2), clr, 2)

    @staticmethod
    def _draw_hud(frame, angle, direction, steer, both_hands,
                  face_ok, mouth_open, mar, fps, profile_name,
                  horn, ind_l, ind_r):
        h, w = frame.shape[:2]
        f    = cv2.FONT_HERSHEY_SIMPLEX
        ov   = frame.copy()
        cv2.rectangle(ov, (0, h-155), (w, h), (10,10,20), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        bw  = int(w * 0.55); bh = 16; bx = (w - bw) // 2; by = h - 115; mid = bx + bw // 2
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40,40,55), -1)
        cv2.rectangle(frame, (mid-2, by-4), (mid+2, by+bh+4), (180,180,180), -1)
        fill = int((bw // 2) * abs(steer))
        if steer < -0.01 and fill:
            cv2.rectangle(frame, (mid-fill, by), (mid,      by+bh), (60,120,255), -1)
        elif steer > 0.01 and fill:
            cv2.rectangle(frame, (mid,      by), (mid+fill, by+bh), (50,220,140), -1)

        dc = (60,120,255) if direction=='LEFT' else ((50,220,140) if direction=='RIGHT' else (200,200,200))
        cv2.putText(frame, ' <- LEFT',       (bx, by-10),        f, 0.45, (60,120,255), 1)
        cv2.putText(frame, 'RIGHT ->',       (bx+bw-80, by-10),  f, 0.45, (50,220,140), 1)
        cv2.putText(frame, f'{steer:+.2f}',  (mid-30, by+bh+28), f, 0.75, dc,           2)
        if SHOW_ANGLE:
            cv2.putText(frame, f'tilt {angle:+.1f}', (bx, h-68), f, 0.45, (255,255,255), 1)
        cv2.putText(frame, f'[{profile_name}]', (bx+bw-120, h-68), f, 0.48, (0,180,255), 1)

        tc = (50,80,255) if mouth_open else (50,220,140)
        tl = 'BRAKE' if mouth_open else 'GAS'
        if face_ok: cv2.putText(frame, f'{tl} {mar:.2f}', (10, h-68), f, 0.47, tc, 1)
        else:       cv2.putText(frame, 'NO FACE',          (10, h-68), f, 0.47, (100,100,100), 1)

        lc = (60,120,255) if ind_l else (50,50,60)
        rc = (50,220,140) if ind_r else (50,50,60)
        cv2.putText(frame, '<<', (10,   60), f, 0.9, lc, 2 if ind_l else 1)
        cv2.putText(frame, '>>', (w-60, 60), f, 0.9, rc, 2 if ind_r else 1)
        if horn: cv2.putText(frame, 'HORN!', (w//2-38, 60), f, 0.85, (0,220,255), 2)

        cv2.putText(frame, f'FPS:{fps:.0f}', (w-75, 30), f, 0.5, (0,180,255), 1)
        hs = 'HANDS OK' if both_hands else 'SHOW BOTH HANDS'
        cv2.putText(frame, hs, (10, 30), f, 0.5, (60,220,60) if both_hands else (0,80,255), 1)

        CameraWorker._draw_wheel(frame, mid, h-32, steer, angle)

    # ── main thread ───────────────────────────────────────────────────────────
    def _run(self):
        if not _CAM_LIBS:
            self.error = 'MediaPipe / OpenCV not installed.\nRun: pip install mediapipe==0.10.14 opencv-python'
            return
        if not _GAMEPAD:
            self.error = ('vgamepad not installed or ViGEm Bus Driver missing.\n'
                          '1. Install ViGEm Bus Driver: https://github.com/nefarius/ViGEmBus/releases\n'
                          '2. pip install vgamepad')
            return

        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
        try:
            cam_idx, _ = _find_camera(backend, preferred=CAMERA_INDEX)
        except RuntimeError as e:
            self.error = str(e); return

        cam  = _CamStream(cam_idx, backend).start()
        ctrl = _CamCtrl()
        tpe  = ThreadPoolExecutor(max_workers=2)

        hands = _mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF)
        face  = _mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF)
        pt_s = _mp_draw.DrawingSpec(color=(200,200,255), thickness=1, circle_radius=2)
        ln_s = _mp_draw.DrawingSpec(color=(80,80,100),  thickness=1)

        speed_key = ord('1')
        angle = 0.0; direction = 'STRAIGHT'; steer = 0.0; mar = 0.0
        lost = 0; face_ok = False; prev_t = time.perf_counter(); dead = 0

        print('[CAM] Camera mode started. Click Stop to quit.')

        while not self._stop.is_set():
            ret, raw = cam.get()
            if not ret or raw is None:
                dead += 1
                if dead > 90: self.error = 'Camera feed lost. Unplug/replug and try again.'; break
                time.sleep(0.01); continue
            dead = 0
            try:
                frame = raw.copy()
                if FLIP_CAMERA: frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                ph, pw = int(h * PROCESS_SCALE), int(w * PROCESS_SCALE)
                small  = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_NEAREST)
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                fh = tpe.submit(hands.process, rgb)
                ff = tpe.submit(face.process,  rgb)
                try:
                    hr = fh.result(timeout=0.15)
                    fr = ff.result(timeout=0.15)
                except Exception:
                    class _R:
                        multi_hand_landmarks = None
                        multi_face_landmarks = None
                    hr = fr = _R()

                now     = time.perf_counter()
                profile = _CAM_PROFILES.get(speed_key, _CAM_PROFILES[ord('1')])

                # HANDS
                both_visible = False
                if hr.multi_hand_landmarks and len(hr.multi_hand_landmarks) >= 2:
                    detected = sorted(hr.multi_hand_landmarks, key=lambda lm: lm.landmark[0].x)
                    lm_l, lm_r = detected[0], detected[1]
                    for lm in (lm_l, lm_r):
                        _mp_draw.draw_landmarks(frame, lm, _mp_hands.HAND_CONNECTIONS, pt_s, ln_s)
                    lx, ly = lm_l.landmark[0].x, lm_l.landmark[0].y
                    rx, ry = lm_r.landmark[0].x, lm_r.landmark[0].y
                    lpx, lpy = int(lx*w), int(ly*h)
                    rpx, rpy = int(rx*w), int(ry*h)
                    cv2.line(frame, (lpx, lpy), (rpx, rpy), (30,100,200), 8)
                    cv2.line(frame, (lpx, lpy), (rpx, rpy), (0,180,255), 2)
                    cv2.circle(frame, (lpx, lpy), 10, (255,130,60),  -1)
                    cv2.circle(frame, (rpx, rpy), 10, (60, 230,130), -1)
                    both_visible = True; lost = 0
                    angle, direction = ctrl.update_steering((lx,ly), (rx,ry), profile['soft_zone'])
                    steer = ctrl.steer_axis
                else:
                    lost += 1

                if not both_visible and lost >= GRACE_FRAMES:
                    ctrl.ease_to_center()
                    steer = ctrl.steer_axis; direction = 'STRAIGHT'

                # FACE
                face_ok = False
                if fr.multi_face_landmarks:
                    lms     = fr.multi_face_landmarks[0].landmark
                    face_ok = True
                    mar     = _mouth_ar(lms, w, h)
                    ctrl.update_throttle(mar)
                    ctrl.update_horn(_brow_raise(lms))
                    ctrl.update_indicators(_head_tilt(lms), now)
                else:
                    ctrl.release_throttle()
                    if ctrl._horn_held: ctrl.kb.release(KEY_HORN); ctrl._horn_held = False

                fps = 1.0 / max(now - prev_t, 1e-9)
                prev_t = now

                self._draw_hud(frame, angle, direction, steer, both_visible,
                               face_ok, ctrl.mouth_open, mar, fps, profile['name'],
                               ctrl.horn_held, ctrl.ind_left, ctrl.ind_right)

                # Push to display queue — always keep the freshest frame
                try:
                    self._fq.put_nowait(frame)
                except queue.Full:
                    try: self._fq.get_nowait()
                    except queue.Empty: pass
                    try: self._fq.put_nowait(frame)
                    except queue.Full: pass

                self._state['steer'] = steer

            except Exception as exc:
                print(f'[CAM] Frame skipped: {exc}')

        ctrl.release_all()
        tpe.shutdown(wait=False)
        hands.close(); face.close(); cam.stop()
        print('[CAM] Camera mode stopped.')


# ════════════════════════════════════════════════════════════════════════════════
# PHONE CONTROLLER
# ════════════════════════════════════════════════════════════════════════════════

class _PhoneCtrl:
    """Translates phone WebSocket messages → gamepad/keyboard inputs."""

    def __init__(self):
        self.kb = _KbCtrl() if _KB else None
        self._ema    = 0.0
        self._steer  = 0.0
        self._key_up = False; self._key_dn = False
        self._horn_held = False
        self._prev_su = False; self._prev_sd = False
        self._cam_idx = 0

        if _GAMEPAD:
            self.pad = vg.VX360Gamepad()
            print('[PHONE] Virtual Xbox gamepad ready.')
        else:
            self.pad = None
            print('[PHONE] vgamepad not available — keyboard fallback only.')

    def update(self, tilt, gas, brake, horn, profile,
               shift_up, shift_down, ind_left, ind_right, lights,
               engine_pulse, park_pulse, cruise_pulse, diff_pulse,
               hazard_pulse, cam_cycle, reverse_pulses):
        soft_zone   = _PHONE_PROFILES.get(profile, _PHONE_PROFILES[1])['soft_zone']
        self._ema   = EMA_ALPHA * tilt + (1 - EMA_ALPHA) * self._ema
        self._steer = tilt_to_axis(self._ema, soft_zone)

        if self.pad:
            try:
                # Negate: phone tilt right → positive → right turn in ETS2
                self.pad.left_joystick_float(x_value_float=-self._steer, y_value_float=0.0)
                self.pad.update()
            except Exception:
                pass

        if self.kb:
            # ── throttle ──────────────────────────────────────────────────────
            if gas:     self._kp(_Key.up);   self._kr(_Key.down)
            elif brake: self._kp(_Key.down); self._kr(_Key.up)
            else:       self._kr(_Key.up);   self._kr(_Key.down)

            # ── horn ──────────────────────────────────────────────────────────
            if horn and not self._horn_held:
                self.kb.press(KEY_HORN);   self._horn_held = True
            elif not horn and self._horn_held:
                self.kb.release(KEY_HORN); self._horn_held = False

            # ── gear shifts ───────────────────────────────────────────────────
            if shift_up   and not self._prev_su: self._shift(True)
            if shift_down and not self._prev_sd: self._shift(False)
            self._prev_su = shift_up; self._prev_sd = shift_down

            # ── reverse: N rapid shift-down pulses to reach reverse ───────────
            if reverse_pulses > 0:
                for _ in range(min(int(reverse_pulses), 14)):
                    self._shift(False)

            # ── indicators / lights ───────────────────────────────────────────
            if ind_left:  self.kb.press(KEY_IND_LEFT);  self.kb.release(KEY_IND_LEFT)
            if ind_right: self.kb.press(KEY_IND_RIGHT); self.kb.release(KEY_IND_RIGHT)
            if lights:    self.kb.press(KEY_LIGHTS);    self.kb.release(KEY_LIGHTS)

            # ── one-shot feature taps ─────────────────────────────────────────
            if engine_pulse: self.kb.press(KEY_ENGINE);     self.kb.release(KEY_ENGINE)
            if cruise_pulse: self.kb.press(KEY_CRUISE);     self.kb.release(KEY_CRUISE)
            if hazard_pulse: self.kb.press(KEY_HAZARD);     self.kb.release(KEY_HAZARD)
            if park_pulse:   self.kb.press(KEY_PARK_BRAKE); self.kb.release(KEY_PARK_BRAKE)
            if diff_pulse:   self.kb.press(KEY_DIFF_LOCK);  self.kb.release(KEY_DIFF_LOCK)

            # ── camera cycle ──────────────────────────────────────────────────
            if cam_cycle:
                self._cam_idx = (self._cam_idx + 1) % len(KEY_CAM_VIEWS)
                k = KEY_CAM_VIEWS[self._cam_idx]
                self.kb.press(k); self.kb.release(k)

    def _shift(self, up: bool):
        if not self.pad: return
        btn = (vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER if up
               else vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        try:
            self.pad.press_button(button=btn); self.pad.update()
            time.sleep(0.05)
            self.pad.release_button(button=btn); self.pad.update()
        except Exception:
            pass

    def _kp(self, key):
        if not self.kb: return
        if key == _Key.up   and not self._key_up: self.kb.press(_Key.up);   self._key_up = True
        if key == _Key.down and not self._key_dn: self.kb.press(_Key.down); self._key_dn = True

    def _kr(self, key):
        if not self.kb: return
        if key == _Key.up   and self._key_up: self.kb.release(_Key.up);   self._key_up = False
        if key == _Key.down and self._key_dn: self.kb.release(_Key.down); self._key_dn = False

    def release_all(self):
        self._kr(_Key.up); self._kr(_Key.down)
        if self.kb and self._horn_held: self.kb.release(KEY_HORN); self._horn_held = False
        if self.pad:
            try:
                self.pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.pad.update()
            except Exception: pass

    @property
    def steer_axis(self) -> float: return self._steer


# ════════════════════════════════════════════════════════════════════════════════
# PHONE SERVER — HTML (phone UI served over HTTPS)
# ════════════════════════════════════════════════════════════════════════════════

_PHONE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Wheel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;touch-action:manipulation}
html,body{width:100%;height:100%;overflow:hidden;background:#07070f;color:#fff;font-family:Arial,sans-serif;user-select:none}
#rot{position:fixed;inset:0;background:#07070f;z-index:300;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px}
#rot span{font-size:52px;animation:spin 2s ease-in-out infinite}
#rot p{color:#555;font-size:14px;letter-spacing:1px}
@keyframes spin{0%,100%{transform:rotate(0deg)}50%{transform:rotate(90deg)}}
@media(orientation:landscape){#rot{display:none}}
#tap{position:fixed;inset:0;background:#07070fdd;z-index:200;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;cursor:pointer}
#tap h2{font-size:20px;letter-spacing:3px;color:#00c8ff}
#tap p{color:#444;font-size:12px;text-align:center;line-height:1.8}
#pulse{width:52px;height:52px;border-radius:50%;border:2px solid #00c8ff;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(.85);opacity:.4}50%{transform:scale(1.1);opacity:1}}
#tap.hide{display:none}
#otp{position:fixed;inset:0;background:#07070f;z-index:150;display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px}
#otp.show{display:flex}
#otp h2{font-size:18px;letter-spacing:3px;color:#00c8ff}
#otp p{color:#555;font-size:12px;text-align:center;line-height:1.7}
#otp-inp{background:#0e0e1c;border:2px solid #222;border-radius:8px;color:#fff;font-size:28px;letter-spacing:8px;text-align:center;width:200px;padding:10px;outline:none;-webkit-appearance:none}
#otp-inp:focus{border-color:#00c8ff}
#otp-btn{background:#091520;border:2px solid #00c8ff;color:#00c8ff;font-size:14px;font-weight:bold;letter-spacing:2px;padding:11px 36px;border-radius:8px;cursor:pointer}
#otp-err{color:#ff4060;font-size:12px;min-height:18px;text-align:center}
#app{display:none;width:100%;height:100%;flex-direction:column}
#app.auth{display:flex}
#st{display:flex;align-items:center;padding:3px 8px;background:#0e0e1c;flex-shrink:0;height:26px;gap:6px}
#dot{width:8px;height:8px;border-radius:50%;background:#333;display:inline-block;transition:background .3s;flex-shrink:0}
#dot.ok{background:#00e676}#dot.err{background:#f44}
#lbl{font-size:10px;color:#666;flex:1;min-width:0;overflow:hidden;white-space:nowrap}
#camb-st{padding:2px 8px;border:1px solid #222;border-radius:3px;background:none;color:#555;font-size:10px;cursor:pointer;transition:all .15s}
#profs{display:flex;gap:3px}
.pb{padding:2px 7px;border:1px solid #222;border-radius:3px;background:none;color:#555;font-size:10px;cursor:pointer;transition:all .15s}
.pb.on{border-color:#00c8ff;color:#00c8ff;background:#091520}
#body{flex:1;display:flex;min-height:0;gap:4px;padding:4px}
#lcol{width:22%;display:flex;flex-direction:column;gap:3px}
#rcol{width:22%;display:flex;flex-direction:column;gap:3px}
#ccol{flex:1;display:flex;flex-direction:column;gap:3px;min-width:0}
.side-sm{flex-shrink:0;border:none;border-radius:7px;background:#0f0f20;color:#555;font-size:10px;font-weight:bold;letter-spacing:.5px;padding:0;height:26px;cursor:pointer;touch-action:none;transition:all .12s}
.side-sm.flash,#ind-l.flash,#ind-r.flash{color:#ffcc00;background:#1a1500;border:1px solid #ffcc00}
.shiftb{flex-shrink:0;border:none;border-radius:7px;background:#0f0f20;color:#777;font-size:12px;font-weight:bold;height:33px;cursor:pointer;touch-action:none;letter-spacing:1px;transition:filter .1s}
.shiftb.on{filter:brightness(2.2);color:#fff}
.bigb{flex:1;border:none;border-radius:10px;font-size:16px;font-weight:bold;letter-spacing:2px;cursor:pointer;display:flex;align-items:center;justify-content:center;touch-action:none;transition:filter .1s;min-height:0}
#bkb{background:linear-gradient(170deg,#4a0000,#991200);color:#ffaaaa}
#gsb{background:linear-gradient(170deg,#002800,#007515);color:#aaffaa}
.bigb.on{filter:brightness(1.6)}
.togb{flex-shrink:0;border:1px solid #1a1a30;border-radius:7px;background:#0a0a18;color:#555;font-size:10px;font-weight:bold;height:26px;cursor:pointer;touch-action:none;letter-spacing:.3px;transition:all .15s}
.togb.on{border-color:#00e676;color:#00e676;background:#001a0c}
#pkb.on{border-color:#ffcc00;color:#ffcc00;background:#1a1200}
#difb.on{border-color:#ff8c00;color:#ff8c00;background:#180d00}
#trow{flex-shrink:0;display:flex;align-items:center;gap:6px;padding:0 2px}
#tbg{flex:1;height:5px;background:#111120;border-radius:3px;position:relative}
#tfill{position:absolute;top:0;height:100%;border-radius:3px;transition:all .04s}
#tctr{position:absolute;left:50%;top:-3px;width:2px;height:11px;background:#333;transform:translateX(-50%)}
#tval{min-width:44px;text-align:right;font-size:10px;color:#00c8ff;font-weight:bold}
.centb{flex:1;border:none;border-radius:9px;font-size:12px;font-weight:bold;cursor:pointer;touch-action:none;letter-spacing:.5px;transition:all .12s;min-height:0;display:flex;align-items:center;justify-content:center}
#revb{background:#12061e;color:#aa77ee;border:1px solid #2a0c44}
#revb.on{background:#220840;color:#dd99ff;border-color:#7733bb}
#hazb{background:#151000;color:#776600;border:1px solid #302200}
#hazb.on{background:#1a1200;color:#ffcc00;border-color:#bb9900;animation:hazf .45s steps(1) infinite}
@keyframes hazf{0%{opacity:1}50%{opacity:.3}}
#hlrow{flex-shrink:0;display:grid;grid-template-columns:1fr 1fr;gap:3px}
.hlb{border:none;border-radius:7px;background:#0f0f20;color:#555;font-size:11px;font-weight:bold;padding:5px 3px;cursor:pointer;letter-spacing:.5px;touch-action:none;transition:all .12s;height:30px}
#hornb.on{color:#00c8ff;background:#001820}
#lightsb.on{color:#ffe066;background:#1a1400}
.hlb.flash{color:#fff;background:#1a1a30}
</style>
</head>
<body>
<div id="rot"><span>📱</span><p>ROTATE TO LANDSCAPE</p></div>
<div id="tap">
  <div id="pulse"></div>
  <h2>🎮 STEERING WHEEL</h2>
  <p>Tap to enable motion sensors,<br>then enter the OTP shown on your PC</p>
</div>
<div id="otp">
  <h2>🔐 ENTER OTP</h2>
  <p>Type the 6-digit code shown<br>on the desktop app</p>
  <input id="otp-inp" type="tel" maxlength="6" inputmode="numeric" placeholder="● ● ● ● ● ●" autocomplete="off">
  <button id="otp-btn">▶ CONNECT</button>
  <div id="otp-err"></div>
</div>
<div id="app">
  <div id="st">
    <span id="dot"></span><span id="lbl">Waiting…</span>
    <button id="camb-st" ontouchstart="tapCam()" onmousedown="tapCam()">📷 CAM</button>
    <div id="profs">
      <button class="pb on" data-p="1">CITY</button>
      <button class="pb"    data-p="2">HWY</button>
      <button class="pb"    data-p="3">MWAY</button>
    </div>
  </div>
  <div id="body">
    <div id="lcol">
      <button class="side-sm" id="ind-l" ontouchstart="tapInd('l')" onmousedown="tapInd('l')">◄ IND</button>
      <button class="shiftb" id="sdnb" ontouchstart="tapShift('d')" onmousedown="tapShift('d')">▼ SHIFT</button>
      <button class="bigb" id="bkb"
        ontouchstart="sb2('b',true)" ontouchend="sb2('b',false)"
        onmousedown="sb2('b',true)" onmouseup="sb2('b',false)">BRAKE</button>
      <button class="togb" id="engb" ontouchstart="tapEngine()" onmousedown="tapEngine()">⚙ ENGINE</button>
      <button class="togb" id="pkb" ontouchstart="tapPark()" onmousedown="tapPark()">🅿 PARK</button>
    </div>
    <div id="ccol">
      <div id="trow">
        <div id="tbg"><div id="tfill"></div><div id="tctr"></div></div>
        <div id="tval">0.0°</div>
      </div>
      <button class="centb" id="revb" ontouchstart="tapReverse()" onmousedown="tapReverse()">◀ REVERSE</button>
      <button class="centb" id="hazb" ontouchstart="tapHazard()" onmousedown="tapHazard()">⚠ HAZARD</button>
      <div id="hlrow">
        <button class="hlb" id="hornb"
          ontouchstart="sh(true)" ontouchend="sh(false)"
          onmousedown="sh(true)" onmouseup="sh(false)">📯 HORN</button>
        <button class="hlb" id="lightsb"
          ontouchstart="tapLights()" onmousedown="tapLights()">💡 LIGHTS</button>
      </div>
    </div>
    <div id="rcol">
      <button class="side-sm" id="ind-r" ontouchstart="tapInd('r')" onmousedown="tapInd('r')">IND ►</button>
      <button class="shiftb" id="supb" ontouchstart="tapShift('u')" onmousedown="tapShift('u')">▲ SHIFT</button>
      <button class="bigb" id="gsb"
        ontouchstart="sb2('g',true)" ontouchend="sb2('g',false)"
        onmousedown="sb2('g',true)" onmouseup="sb2('g',false)">GAS</button>
      <button class="togb" id="ccb" ontouchstart="tapCruise()" onmousedown="tapCruise()">🚗 CRUISE</button>
      <button class="togb" id="difb" ontouchstart="tapDiff()" onmousedown="tapDiff()">🔒 DIFF</button>
    </div>
  </div>
</div>
<script>
const WS_PORT=__WS_PORT__;
let tilt=0,gas=false,brake=false,horn=false,profile=1;
let shiftUp=false,shiftDown=false,indLeft=false,indRight=false,lights=false;
let enginePulse=false,parkPulse=false,cruisePulse=false,diffPulse=false,hazardPulse=false;
let camCycle=false,reversePulses=0;
let localGear=0,_pu=false,_pd=false;
let engineOn=false,parkOn=false,cruiseOn=false,diffOn=false,hazardOn=false,revOn=false;
let _camView=0,_otp='';
let ws,sendLoop;
const dot=document.getElementById('dot'),lbl=document.getElementById('lbl');
// ── Audio ─────────────────────────────────────────────────────────────────────
let _ac=null;
function ac(){if(!_ac)try{_ac=new(window.AudioContext||window.webkitAudioContext)()}catch(e){}; return _ac;}
function tone(f,t,v,tp,at){
  const a=ac();if(!a)return;
  const now=at!==undefined?at:a.currentTime;
  const o=a.createOscillator(),g=a.createGain();
  o.connect(g);g.connect(a.destination);
  o.type=tp||'sine';o.frequency.value=f;
  g.gain.setValueAtTime(0,now);g.gain.linearRampToValueAtTime(v,now+0.008);
  g.gain.setValueAtTime(v,now+t-0.03);g.gain.linearRampToValueAtTime(0,now+t);
  o.start(now);o.stop(now+t);
}
function click(){tone(300,0.05,0.15,'square');}
function shiftSnd(up){tone(up?520:260,0.08,0.25,'triangle');}
function hornSnd(){tone(350,0.15,0.4,'sawtooth');}
function engStartSnd(){const a=ac();if(!a)return;const n=a.currentTime;tone(100,0.45,0.35,'sawtooth',n);tone(180,0.25,0.25,'sawtooth',n+0.3);}
function engStopSnd(){const a=ac();if(!a)return;const n=a.currentTime;tone(160,0.12,0.3,'sawtooth',n);tone(80,0.35,0.2,'sawtooth',n+0.1);}
function hazSnd(){tone(700,0.07,0.3,'sine');setTimeout(()=>tone(700,0.07,0.3,'sine'),140);}
// ── Reverse beep ──────────────────────────────────────────────────────────────
let _revTimer=null;
function _revBeep(){tone(880,0.1,0.45,'sine');}
function startRevBeep(){if(_revTimer)return;_revBeep();_revTimer=setInterval(_revBeep,750);}
function stopRevBeep(){clearInterval(_revTimer);_revTimer=null;}
// ── Vibration ─────────────────────────────────────────────────────────────────
function vib(p){try{navigator.vibrate&&navigator.vibrate(p);}catch(e){}}
// ── Gear tracking ─────────────────────────────────────────────────────────────
function trackGear(){
  if(shiftUp&&!_pu)  localGear=Math.min(12,localGear+1);
  if(shiftDown&&!_pd) localGear=Math.max(-1,localGear-1);
  _pu=shiftUp;_pd=shiftDown;
}
// ── Button handlers ───────────────────────────────────────────────────────────
function sb2(t,s){
  if(t==='g'){gas=s;document.getElementById('gsb').classList.toggle('on',s);}
  if(t==='b'){brake=s;document.getElementById('bkb').classList.toggle('on',s);}
}
function sh(s){horn=s;document.getElementById('hornb').classList.toggle('on',s);if(s){hornSnd();vib([80,60,80]);}}
function tapShift(d){
  if(d==='u'){shiftUp=true;flash('supb');shiftSnd(true);vib(55);}
  if(d==='d'){shiftDown=true;flash('sdnb');shiftSnd(false);vib(55);}
  sendNow();shiftUp=false;shiftDown=false;
}
function tapInd(s){
  if(s==='l'){indLeft=true;flash('ind-l');}else{indRight=true;flash('ind-r');}
  click();vib(45);sendNow();indLeft=false;indRight=false;
}
function tapLights(){lights=true;flash('lightsb');click();vib(55);sendNow();lights=false;}
function tapEngine(){
  engineOn=!engineOn;enginePulse=true;
  document.getElementById('engb').classList.toggle('on',engineOn);
  if(engineOn){engStartSnd();vib([80,40,180]);}else{engStopSnd();vib(120);}
  sendNow();enginePulse=false;
}
function tapPark(){
  parkOn=!parkOn;parkPulse=true;
  document.getElementById('pkb').classList.toggle('on',parkOn);
  click();vib(90);sendNow();parkPulse=false;
}
function tapCruise(){
  cruiseOn=!cruiseOn;cruisePulse=true;
  document.getElementById('ccb').classList.toggle('on',cruiseOn);
  click();vib(70);sendNow();cruisePulse=false;
}
function tapDiff(){
  diffOn=!diffOn;diffPulse=true;
  document.getElementById('difb').classList.toggle('on',diffOn);
  click();vib([55,35,55]);sendNow();diffPulse=false;
}
function tapHazard(){
  hazardOn=!hazardOn;hazardPulse=true;
  document.getElementById('hazb').classList.toggle('on',hazardOn);
  hazSnd();vib([45,45,45,45,45]);sendNow();hazardPulse=false;
}
function tapReverse(){
  if(!revOn){
    reversePulses=localGear+1;
    localGear=-1;revOn=true;
    document.getElementById('revb').classList.add('on');
    startRevBeep();vib([70,35,70,35,70]);
    sendNow();reversePulses=0;
  } else {
    revOn=false;localGear=0;
    document.getElementById('revb').classList.remove('on');
    stopRevBeep();vib(80);
    shiftUp=true;sendNow();shiftUp=false;
  }
}
function tapCam(){
  _camView=(_camView+1)%3;camCycle=true;click();vib(45);
  sendNow();camCycle=false;
}
document.querySelectorAll('.pb').forEach(b=>b.addEventListener('click',()=>{
  profile=+b.dataset.p;
  document.querySelectorAll('.pb').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');click();
}));
function flash(id){const e=document.getElementById(id);e.classList.add('flash');setTimeout(()=>e.classList.remove('flash'),220);}
// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect(){
  ws=new WebSocket('wss://'+window.location.hostname+':'+WS_PORT);
  ws.onopen=()=>{ws.send(JSON.stringify({otp:_otp}));};
  ws.onmessage=e=>{
    try{
      const d=JSON.parse(e.data);
      if(d.otpOk){
        dot.className='ok';lbl.textContent='Connected ✓';
        document.getElementById('app').classList.add('auth');
        sendLoop=setInterval(sendCtrl,33);
      } else if(d.otpErr){
        ws.close();
        document.getElementById('otp-err').textContent='❌ Wrong code — check the desktop app';
        document.getElementById('otp').classList.add('show');
      }
    }catch(err){}
  };
  ws.onclose=ws.onerror=()=>{
    dot.className='err';lbl.textContent='Reconnecting…';
    clearInterval(sendLoop);stopRevBeep();
    if(document.getElementById('app').classList.contains('auth'))
      setTimeout(()=>document.getElementById('otp').classList.add('show'),1500);
  };
}
document.getElementById('otp-btn').addEventListener('click',doOtp);
document.getElementById('otp-btn').addEventListener('touchend',e=>{e.preventDefault();doOtp();});
document.getElementById('otp-inp').addEventListener('keydown',e=>{if(e.key==='Enter')doOtp();});
function doOtp(){
  _otp=document.getElementById('otp-inp').value.trim();
  if(_otp.length<4){document.getElementById('otp-err').textContent='Enter the code from your PC';return;}
  document.getElementById('otp-err').textContent='Connecting…';
  document.getElementById('otp').classList.remove('show');
  dot.className='';lbl.textContent='Authenticating…';
  connect();
}
function buildMsg(){
  return{tilt,gas,brake,horn,profile,
         shiftUp,shiftDown,indLeft,indRight,lights,
         enginePulse,parkPulse,cruisePulse,diffPulse,hazardPulse,
         camCycle,reversePulses};
}
function sendNow(){if(ws&&ws.readyState===1)ws.send(JSON.stringify(buildMsg()));}
function sendCtrl(){
  if(ws&&ws.readyState===1)ws.send(JSON.stringify(buildMsg()));
  indLeft=indRight=lights=shiftUp=shiftDown=false;
  enginePulse=parkPulse=cruisePulse=diffPulse=hazardPulse=camCycle=false;
  reversePulses=0;
}
// ── Tilt ──────────────────────────────────────────────────────────────────────
function getLandscapeAngle(){
  if(screen.orientation&&screen.orientation.angle!=null)return screen.orientation.angle;
  const w=window.orientation||0;return w===-90?90:w===90?270:0;
}
function onOri(e){
  const a=getLandscapeAngle();
  let raw=a===90?-(e.beta||0):a===270?(e.beta||0):(e.gamma||0);
  tilt=Math.max(-90,Math.min(90,raw));
}
const tfill=document.getElementById('tfill'),tvalEl=document.getElementById('tval');
function updateTilt(){
  const abs=Math.abs(tilt);
  tvalEl.textContent=(tilt>=0?'+':'')+tilt.toFixed(1)+'°';
  if(tilt<-1)      tfill.style.cssText=`left:${50-abs/90*50}%;width:${abs/90*50}%;background:#4488ff`;
  else if(tilt>1)  tfill.style.cssText=`left:50%;width:${abs/90*50}%;background:#00e676`;
  else             tfill.style.cssText='width:0';
}
function frame(){trackGear();updateTilt();requestAnimationFrame(frame);}
const tapEl=document.getElementById('tap');
async function start(){
  if(typeof DeviceOrientationEvent!=='undefined'&&typeof DeviceOrientationEvent.requestPermission==='function'){
    try{const r=await DeviceOrientationEvent.requestPermission();if(r!=='granted'){alert('Sensor access denied — enable Motion in Safari Settings.');return;}}catch(e){}
  }
  window.addEventListener('deviceorientation',onOri,{passive:true});
  tapEl.classList.add('hide');
  document.getElementById('otp').classList.add('show');
  frame();
}
tapEl.addEventListener('click',start);
tapEl.addEventListener('touchend',e=>{e.preventDefault();start();});
window.addEventListener('orientationchange',()=>setTimeout(()=>{},150));
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════════════
# PHONE SERVER  (HTTPS + WSS, runs in asyncio thread)
# ════════════════════════════════════════════════════════════════════════════════

def _ensure_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    print('[CERT] Generating self-signed certificate (first run only)…')
    try:
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048',
                        '-keyout',KEY_FILE,'-out',CERT_FILE,'-days','730',
                        '-nodes','-subj','/CN=SteeringWheel'],
                       check=True, capture_output=True)
        print('[CERT] Done (openssl).'); return
    except Exception:
        pass
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'SteeringWheel')])
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.utcnow())
                .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=730))
                .sign(key, hashes.SHA256()))
        with open(KEY_FILE, 'wb') as fh:
            fh.write(key.private_bytes(serialization.Encoding.PEM,
                     serialization.PrivateFormat.TraditionalOpenSSL,
                     serialization.NoEncryption()))
        with open(CERT_FILE, 'wb') as fh:
            fh.write(cert.public_bytes(serialization.Encoding.PEM))
        print('[CERT] Done (cryptography).')
    except ImportError:
        print('[ERROR] pip install cryptography')


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '127.0.0.1'


class PhoneServer:
    """Manages the HTTPS page server and WSS control channel for phone mode."""

    def __init__(self, on_connect=None, on_disconnect=None, steer_ref: dict = None):
        self._on_connect    = on_connect
        self._on_disconnect = on_disconnect
        self._steer_ref     = steer_ref
        self._ctrl          = _PhoneCtrl()
        self._clients: set  = set()
        self._lock          = threading.Lock()
        self._ssl_ctx       = None
        self._http_srv      = None
        self.ip             = ''
        self.otp            = ''

    def start(self, loop: asyncio.AbstractEventLoop):
        self.otp = str(random.randint(100000, 999999))
        _ensure_cert()
        self._ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)

        ssl_ctx = self._ssl_ctx

        class _H(BaseHTTPRequestHandler):
            def do_GET(self):
                b = _PHONE_HTML.replace('__WS_PORT__', str(WS_PORT)).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type',   'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers(); self.wfile.write(b)
            def log_message(self, *_): pass

        self._http_srv = HTTPServer(('0.0.0.0', HTTP_PORT), _H)
        self._http_srv.socket = ssl_ctx.wrap_socket(self._http_srv.socket, server_side=True)
        threading.Thread(target=self._http_srv.serve_forever, daemon=True).start()

        asyncio.run_coroutine_threadsafe(self._ws_serve(), loop)
        self.ip = _local_ip()
        print(f'[PHONE] Server ready.  Open on phone: https://{self.ip}:{HTTP_PORT}')
        print(f'[PHONE] OTP: {self.otp}  (shown in GUI — enter on phone)')
        print('[PHONE] Tap  Advanced → Proceed  to skip the SSL warning.')

    def stop(self):
        try: self._ctrl.release_all()
        except Exception: pass
        if self._http_srv:
            try: self._http_srv.shutdown()
            except Exception: pass

    async def _ws_serve(self):
        if not _WS:
            print('[PHONE] websockets not installed — pip install websockets'); return
        async with websockets.serve(self._ws_handler, '0.0.0.0', WS_PORT, ssl=self._ssl_ctx):
            await asyncio.Future()   # keep serving forever

    async def _ws_handler(self, ws):
        addr = ws.remote_address[0]
        print(f'[PHONE] Connection attempt: {addr}')

        # ── OTP authentication ────────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            d   = json.loads(raw)
            if str(d.get('otp', '')).strip() != self.otp:
                await ws.send(json.dumps({'otpErr': True}))
                print(f'[PHONE] Rejected bad OTP from {addr}')
                return
        except Exception:
            return
        await ws.send(json.dumps({'otpOk': True}))

        # ── Authenticated ─────────────────────────────────────────────────────
        print(f'[PHONE] Phone authenticated: {addr}')
        with self._lock: self._clients.add(ws)
        if self._on_connect: self._on_connect(addr)
        try:
            async for msg in ws:
                try:
                    d = json.loads(msg)
                    self._ctrl.update(
                        tilt           = float(d.get('tilt',          0)),
                        gas            = bool (d.get('gas',           False)),
                        brake          = bool (d.get('brake',         False)),
                        horn           = bool (d.get('horn',          False)),
                        profile        = int  (d.get('profile',       1)),
                        shift_up       = bool (d.get('shiftUp',       False)),
                        shift_down     = bool (d.get('shiftDown',     False)),
                        ind_left       = bool (d.get('indLeft',       False)),
                        ind_right      = bool (d.get('indRight',      False)),
                        lights         = bool (d.get('lights',        False)),
                        engine_pulse   = bool (d.get('enginePulse',   False)),
                        park_pulse     = bool (d.get('parkPulse',     False)),
                        cruise_pulse   = bool (d.get('cruisePulse',   False)),
                        diff_pulse     = bool (d.get('diffPulse',     False)),
                        hazard_pulse   = bool (d.get('hazardPulse',   False)),
                        cam_cycle      = bool (d.get('camCycle',      False)),
                        reverse_pulses = int  (d.get('reversePulses', 0)),
                    )
                    if self._steer_ref is not None:
                        self._steer_ref['steer'] = self._ctrl.steer_axis
                except Exception as e:
                    print(f'[PHONE] Bad message: {e}')
        except Exception:
            pass
        finally:
            with self._lock: self._clients.discard(ws)
            self._ctrl.release_all()
            if self._on_disconnect: self._on_disconnect(addr)
            print(f'[PHONE] Phone disconnected: {addr}')


# ════════════════════════════════════════════════════════════════════════════════
# GUI LOGGER
# ════════════════════════════════════════════════════════════════════════════════

class _GUIStream:
    """Redirects print() output into a queue that the GUI drains into the log widget."""
    def __init__(self, q: queue.Queue):
        self._q = q
    def write(self, text: str):
        t = text.strip()
        if t: self._q.put(t)
    def flush(self): pass
    def isatty(self): return False


# ════════════════════════════════════════════════════════════════════════════════
# GUI — MAIN APPLICATION WINDOW
# ════════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    # Dark palette
    BG      = '#0d0d1a'
    BG2     = '#09090f'
    PANEL   = '#0f0f1e'
    ACCENT  = '#00c8ff'
    GREEN   = '#00e676'
    RED     = '#ff4060'
    DIM     = '#3a3a5a'
    TEXT    = '#c0c0d8'
    BTN_ON  = '#091520'

    def __init__(self):
        super().__init__()
        self.title('🎮 Virtual Steering Wheel')
        self.configure(bg=self.BG)
        self.geometry('980x680')
        self.minsize(780, 540)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._mode    = 'camera'
        self._running = False
        self._state   = {'steer': 0.0}   # shared steer value for the strip

        self._cam_worker: CameraWorker | None = None
        self._phone_srv:  PhoneServer  | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._log_q:   queue.Queue = queue.Queue()

        self._cam_photo  = None   # PIL PhotoImage reference (prevents GC)
        self._cam_img_id = None   # canvas item id

        self._build_ui()
        self._redirect_output()
        self._poll_log()
        # self._poll_telem()   # telemetry temporarily disabled
        self.after(400, self._auto_detect)

    # ── redirect stdout/stderr ────────────────────────────────────────────────
    def _redirect_output(self):
        s = _GUIStream(self._log_q)
        sys.stdout = s
        sys.stderr = s

    # ══ UI CONSTRUCTION ═══════════════════════════════════════════════════════

    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=self.BG2, pady=7)
        hdr.pack(fill='x')

        tk.Label(hdr, text='🎮 Virtual Steering Wheel',
                 bg=self.BG2, fg=self.ACCENT,
                 font=('Arial', 13, 'bold')).pack(side='left', padx=14)

        mbf = tk.Frame(hdr, bg=self.BG2)
        mbf.pack(side='left', padx=16)
        self._btn_cam   = self._mkbtn(mbf, '📷  Camera', lambda: self._set_mode('camera'))
        self._btn_phone = self._mkbtn(mbf, '📱  Phone',  lambda: self._set_mode('phone'))
        self._btn_cam.pack(side='left', padx=3)
        self._btn_phone.pack(side='left', padx=3)

        self._btn_start = tk.Button(
            hdr, text='▶  START',
            bg='#052505', fg=self.GREEN,
            activebackground='#0a3a0a', activeforeground=self.GREEN,
            font=('Arial', 11, 'bold'), relief='flat', bd=0,
            padx=18, pady=4, cursor='hand2',
            command=self._toggle_start)
        self._btn_start.pack(side='right', padx=14)

        # ── content area ──────────────────────────────────────────────────────
        self._content = tk.Frame(self, bg=self.BG)
        self._content.pack(fill='both', expand=True, padx=8, pady=(4,0))

        self._pane_cam   = self._build_cam_pane(self._content)
        self._pane_phone = self._build_phone_pane(self._content)
        self._pane_cam.pack(fill='both', expand=True)

        # ── telemetry strip ───────────────────────────────────────────────────
        tstrip = tk.Frame(self, bg=self.BG2, pady=5)
        tstrip.pack(fill='x', padx=8, pady=(2,0))
        self._build_telem_strip(tstrip)

        # ── log pane ──────────────────────────────────────────────────────────
        lf = tk.Frame(self, bg=self.BG)
        lf.pack(fill='x', padx=8, pady=(2,6))
        tk.Label(lf, text='System Log', bg=self.BG, fg=self.DIM,
                 font=('Arial', 8)).pack(anchor='w')
        self._log = scrolledtext.ScrolledText(
            lf, height=6, bg='#07070d', fg='#4a4a6a',
            font=('Consolas', 8), state='disabled', relief='flat',
            insertbackground=self.ACCENT, selectbackground=self.DIM)
        self._log.pack(fill='x')

        self._update_mode_btns()

    def _mkbtn(self, parent, text, cmd):
        return tk.Button(parent, text=text, relief='flat', bd=0,
                         font=('Arial', 10), cursor='hand2',
                         padx=12, pady=5, command=cmd)

    def _build_cam_pane(self, parent):
        frame = tk.Frame(parent, bg=self.BG)
        self._cam_cvs = tk.Canvas(frame, bg='#050510', highlightthickness=0)
        self._cam_cvs.pack(fill='both', expand=True)
        self._cam_ph = self._cam_cvs.create_text(
            320, 240,
            text='Camera feed will appear here when running.\n'
                 'Switch to Phone mode if you have no webcam.',
            fill=self.DIM, font=('Arial', 12), anchor='center', justify='center')
        return frame

    def _build_phone_pane(self, parent):
        frame = tk.Frame(parent, bg=self.BG)

        left = tk.Frame(frame, bg=self.BG)
        left.pack(side='left', fill='both', expand=True, padx=24, pady=12)

        tk.Label(left, text='Open this URL on your phone (same Wi-Fi):',
                 bg=self.BG, fg=self.TEXT, font=('Arial', 11)).pack(pady=(12,6))

        self._qr_lbl = tk.Label(left, bg=self.BG)
        self._qr_lbl.pack(pady=6)

        self._url_lbl = tk.Label(left,
                                  text='Start the server to get the URL',
                                  bg=self.BG, fg=self.DIM,
                                  font=('Consolas', 13, 'bold'))
        self._url_lbl.pack()

        tk.Label(left,
                 text='Tap  Advanced → Proceed  to skip the SSL warning.\n'
                      'Safe — the certificate is self-signed and local-only.',
                 bg=self.BG, fg=self.DIM, font=('Arial', 9),
                 justify='center').pack(pady=(8,0))

        self._phone_conn_lbl = tk.Label(left, text='',
                                         bg=self.BG, fg=self.DIM, font=('Arial', 10))
        self._phone_conn_lbl.pack(pady=(12,0))

        # right: info panel
        right = tk.Frame(frame, bg=self.PANEL, padx=16, pady=14)
        right.pack(side='right', fill='y', padx=(0,4), pady=4)

        tk.Label(right, text='Funbit Telemetry Server',
                 bg=self.PANEL, fg=self.DIM, font=('Arial', 9, 'bold')).pack(anchor='w')

        self._telem_live_lbl = tk.Label(right, text='● Waiting for server…',
                                         bg=self.PANEL, fg=self.DIM, font=('Arial', 9))
        self._telem_live_lbl.pack(anchor='w', pady=(2,10))

        info = [
            'Run start_ets2.bat first to launch',
            'ETS2 + the Funbit telemetry server.',
            '',
            'Server URL:',
            '192.168.56.1:25555',
            '',
            'Works with demo & full game.',
        ]
        for line in info:
            tk.Label(right, text=line, bg=self.PANEL,
                     fg=self.DIM if line else self.PANEL,
                     font=('Arial', 8), justify='left').pack(anchor='w')

        return frame

    def _build_telem_strip(self, parent):
        tk.Label(parent, text='Steer:', bg=self.BG2, fg=self.DIM,
                 font=('Arial', 9)).pack(side='left', padx=(6,2))

        self._steer_cvs = tk.Canvas(parent, bg='#0c0c1a', width=220, height=14,
                                     highlightthickness=0)
        self._steer_cvs.pack(side='left')
        self._steer_cvs.create_line(110, 0, 110, 14, fill=self.DIM, width=2)
        self._steer_fill = self._steer_cvs.create_rectangle(
            110, 1, 110, 13, fill=self.ACCENT, outline='')

        tk.Label(parent, text='│', bg=self.BG2, fg=self.DIM).pack(side='left', padx=8)

        self._spd_lbl  = tk.Label(parent, text='Speed: -- ',
                                   bg=self.BG2, fg=self.TEXT, font=('Consolas', 9))
        self._rpm_lbl  = tk.Label(parent, text='RPM: -- ',
                                   bg=self.BG2, fg=self.TEXT, font=('Consolas', 9))
        self._gear_lbl = tk.Label(parent, text='Gear: N ',
                                   bg=self.BG2, fg=self.TEXT, font=('Consolas', 9))
        for w in (self._spd_lbl, self._rpm_lbl, self._gear_lbl):
            w.pack(side='left', padx=3)

        self._live_lbl = tk.Label(parent, text='● SIM',
                                   bg=self.BG2, fg=self.DIM, font=('Arial', 9))
        self._live_lbl.pack(side='right', padx=10)

    # ══ MODE SWITCHING ════════════════════════════════════════════════════════

    def _set_mode(self, mode: str):
        if self._running: self._stop_all()
        self._mode = mode
        if mode == 'camera':
            self._pane_phone.pack_forget()
            self._pane_cam.pack(fill='both', expand=True)
        else:
            self._pane_cam.pack_forget()
            self._pane_phone.pack(fill='both', expand=True)
        self._update_mode_btns()

    def _update_mode_btns(self):
        cam_on = self._mode == 'camera'
        for btn, on in ((self._btn_cam, cam_on), (self._btn_phone, not cam_on)):
            btn.config(bg=self.BTN_ON if on else self.BG2,
                       fg=self.ACCENT  if on else self.DIM,
                       relief='flat')

    # ══ START / STOP ══════════════════════════════════════════════════════════

    def _toggle_start(self):
        if self._running: self._stop_all()
        else:             self._start()

    def _start(self):
        if self._mode == 'camera': self._start_camera()
        else:                      self._start_phone()

    def _start_camera(self):
        if not _CAM_LIBS:
            self._log_q.put('[ERROR] MediaPipe/OpenCV not installed. pip install mediapipe==0.10.14 opencv-python')
            return
        self._cam_worker = CameraWorker(self._frame_q, self._state)
        self._cam_worker.start()
        self._running = True
        self._btn_start.config(text='■  STOP', bg='#280808', fg=self.RED,
                               activebackground='#3a0808', activeforeground=self.RED)
        self._poll_frame()
        self.after(2500, self._check_cam_error)

    def _check_cam_error(self):
        if self._cam_worker and self._cam_worker.error:
            err = self._cam_worker.error
            self._log_q.put(f'[CAM ERROR] {err}')
            self._cam_cvs.itemconfig(self._cam_ph,
                text='⚠ No camera found.\n\nSwitch to Phone Mode to use your phone as a controller.',
                fill=self.RED)
            self._running = False
            self._btn_start.config(text='▶  START', bg='#052505', fg=self.GREEN,
                                   activebackground='#0a3a0a', activeforeground=self.GREEN)

    def _start_phone(self):
        if not _WS:
            self._log_q.put('[ERROR] websockets not installed. pip install websockets')
            return
        self._async_loop = asyncio.new_event_loop()
        threading.Thread(target=self._async_loop.run_forever, daemon=True).start()

        self._phone_srv = PhoneServer(
            on_connect    = lambda ip: self.after(0, lambda: self._phone_conn_lbl.config(
                text=f'✓ Phone connected: {ip}', fg=self.GREEN)),
            on_disconnect = lambda ip: self.after(0, lambda: self._phone_conn_lbl.config(
                text='● Waiting for phone to connect…', fg=self.DIM)),
            steer_ref     = self._state,
        )
        self._phone_srv.start(self._async_loop)

        ip  = self._phone_srv.ip
        url = f'https://{ip}:{HTTP_PORT}'
        self._url_lbl.config(text=url, fg=self.ACCENT)
        self._phone_conn_lbl.config(text='● Waiting for phone to connect…', fg=self.DIM)
        self._show_qr(url)

        self._running = True
        self._btn_start.config(text='■  STOP', bg='#280808', fg=self.RED,
                               activebackground='#3a0808', activeforeground=self.RED)

    def _stop_all(self):
        if self._cam_worker:
            self._cam_worker.stop()
            self._cam_worker = None
        if self._phone_srv:
            self._phone_srv.stop()
            self._phone_srv = None
        if self._async_loop:
            try: self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            except Exception: pass
            self._async_loop = None

        self._running = False
        self._state['steer'] = 0.0
        self._btn_start.config(text='▶  START', bg='#052505', fg=self.GREEN,
                               activebackground='#0a3a0a', activeforeground=self.GREEN)

        # Reset camera canvas
        self._cam_cvs.itemconfig(self._cam_ph,
            text='Camera feed will appear here when running.\n'
                 'Switch to Phone mode if you have no webcam.',
            fill=self.DIM)
        if self._cam_img_id:
            self._cam_cvs.delete(self._cam_img_id)
            self._cam_img_id = None
        self._cam_photo = None

        self._url_lbl.config(text='Start the server to get the URL', fg=self.DIM)
        self._phone_conn_lbl.config(text='', fg=self.DIM)
        self._qr_lbl.config(image='')

    # ══ CAMERA FRAME DISPLAY ══════════════════════════════════════════════════

    def _poll_frame(self):
        if not self._running or self._mode != 'camera':
            return
        try:
            frame = self._frame_q.get_nowait()
            if _PIL and _CAM_LIBS:
                h, w = frame.shape[:2]
                cw = self._cam_cvs.winfo_width()  or w
                ch = self._cam_cvs.winfo_height() or h
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)

                self._cam_cvs.itemconfig(self._cam_ph, text='')
                if self._cam_img_id:
                    self._cam_cvs.coords(self._cam_img_id, cw // 2, ch // 2)
                    self._cam_cvs.itemconfig(self._cam_img_id, image=photo)
                else:
                    self._cam_img_id = self._cam_cvs.create_image(
                        cw // 2, ch // 2, anchor='center', image=photo)
                self._cam_photo = photo   # prevent GC
        except queue.Empty:
            pass
        except Exception as e:
            self._log_q.put(f'[GUI] Frame display error: {e}')

        self.after(30, self._poll_frame)

    # ══ TELEMETRY STRIP UPDATE ════════════════════════════════════════════════

    def _poll_telem(self):
        snap = _telemetry.snapshot
        self._spd_lbl.config(text=f"Speed: {snap['speed']:.0f} km/h")
        self._rpm_lbl.config(text=f"RPM: {snap['rpm']}")
        g = snap['gear']
        self._gear_lbl.config(text=f"Gear: {'N' if g == 0 else ('R' if g < 0 else str(g))}")
        live = snap['live']
        self._live_lbl.config(text='● LIVE' if live else '● SIM (demo)',
                               fg=self.GREEN if live else self.DIM)
        if hasattr(self, '_telem_live_lbl'):
            self._telem_live_lbl.config(
                text=('● Connected — live data' if live else '● Not connected — simulated'),
                fg=self.GREEN if live else self.DIM)

        # Steering bar
        steer = max(-1.0, min(1.0, self._state.get('steer', 0.0)))
        mid   = 110; fill = int(abs(steer) * 108)
        clr   = '#4488ff' if steer < -0.01 else (self.GREEN if steer > 0.01 else self.DIM)
        if steer < -0.01:
            self._steer_cvs.coords(self._steer_fill, mid-fill, 1, mid, 13)
        elif steer > 0.01:
            self._steer_cvs.coords(self._steer_fill, mid, 1, mid+fill, 13)
        else:
            self._steer_cvs.coords(self._steer_fill, mid, 1, mid, 13)
        self._steer_cvs.itemconfig(self._steer_fill, fill=clr)

        self.after(100, self._poll_telem)

    # ══ LOG DRAIN ════════════════════════════════════════════════════════════

    def _poll_log(self):
        while True:
            try:
                text = self._log_q.get_nowait()
                self._log.configure(state='normal')
                self._log.insert('end', text + '\n')
                self._log.see('end')
                # Keep last 200 lines
                lines = int(self._log.index('end-1c').split('.')[0])
                if lines > 200:
                    self._log.delete('1.0', f'{lines-200}.0')
                self._log.configure(state='disabled')
            except queue.Empty:
                break
        self.after(80, self._poll_log)

    # ══ QR CODE ══════════════════════════════════════════════════════════════

    def _show_qr(self, url: str):
        if not (_QRCODE and _PIL):
            return
        try:
            qr = _qr_mod.QRCode(version=1, box_size=5, border=2,
                                 error_correction=_qr_mod.constants.ERROR_CORRECT_L)
            qr.add_data(url); qr.make(fit=True)
            img   = qr.make_image(fill_color='#00c8ff', back_color='#0d0d1a')
            photo = ImageTk.PhotoImage(img)
            self._qr_lbl.config(image=photo)
            self._qr_lbl._photo = photo   # prevent GC
        except Exception as e:
            self._log_q.put(f'[GUI] QR error: {e}')

    # ══ AUTO CAMERA DETECT ════════════════════════════════════════════════════

    def _auto_detect(self):
        if not _CAM_LIBS:
            self._log_q.put('[INFO] MediaPipe/OpenCV not found — defaulting to Phone mode.')
            self._set_mode('phone')
            return

        def _probe():
            try:
                backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
                cap = cv2.VideoCapture(0, backend)
                ok  = cap.isOpened()
                cap.release()
                return ok
            except Exception:
                return False

        def _done(found: bool):
            if found:
                self._log_q.put('[INFO] Camera detected — Camera mode ready. Click START to begin.')
            else:
                self._log_q.put('[INFO] No camera detected — switching to Phone mode.')
                self._set_mode('phone')

        def _thread():
            found = _probe()
            self.after(0, lambda: _done(found))

        threading.Thread(target=_thread, daemon=True).start()

    # ══ CLOSE ════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._stop_all()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
