"""
Virtual Steering Wheel v3 — MediaPipe + vgamepad
=================================================
Gestures
  Hand tilt (both)     →  analog left-stick X  (proportional, speed-adaptive)
  Mouth closed         →  ↑ key  (gas)
  Mouth open           →  ↓ key  (brake)
  Both eyebrows raised →  H key  (horn, held)
  Head tilt left       →  ,      (left indicator toggle)
  Head tilt right      →  .      (right indicator toggle)

Speed profiles  (press 1 / 2 / 3 in the app window)
  1 = CITY      →  full sensitivity  (65 ° = full lock)
  2 = HIGHWAY   →  reduced           (45 ° = full lock)
  3 = MOTORWAY  →  minimal           (28 ° = full lock)
  At high speed the same tilt gives far less steering — like a real car.

Prerequisites
  1. ViGEm Bus Driver  https://github.com/nefarius/ViGEmBus/releases  (free)
  2. pip install vgamepad mediapipe==0.10.14 opencv-python pynput numpy

Run
  py -3.11 steering_wheel.py
"""

import cv2
from mediapipe.python.solutions import hands        as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_module
from mediapipe.python.solutions import face_mesh    as mp_face_mesh_module
import numpy as np
import math, time, threading, platform
from concurrent.futures import ThreadPoolExecutor
from pynput.keyboard import Key, Controller as KeyboardController

try:
    import vgamepad as vg
    _GAMEPAD_AVAILABLE = True
except Exception:
    _GAMEPAD_AVAILABLE = False

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = None    # None = auto-detect (scans 0-4); set to 0/1/2 to force a specific camera
DEAD_ZONE_DEG       = 5       # tilt within this = no steering output
STEERING_CURVE      = 2.5     # 1=linear (harsh), 2.5=exponential curve (gentler at small angles)
FLIP_CAMERA         = True
SHOW_ANGLE          = True
MIN_DETECTION_CONF  = 0.3
MIN_TRACKING_CONF   = 0.2
GRACE_FRAMES        = 30
PROCESS_SCALE       = 0.50    # halved from 0.75 — biggest single perf win

# Smoothing — exponential moving average on the raw angle
EMA_ALPHA           = 0.20    # 0 = max smooth (laggy)  1 = raw (twitchy)  — lowered for steel-plate use

# Re-center — when hands leave frame, axis eases back instead of snapping
RECENTER_RATE       = 0.05    # axis units per frame (0.05 → ~1 s from full lock to centre)

# Speed-adaptive steering: press 1 / 2 / 3 in app window
# soft_zone = hand tilt angle (°) that maps to full lock — larger = less sensitive
SPEED_PROFILES = {
    ord('1'): dict(name="CITY",      soft_zone=70),   # was 40° — much less twitchy in city traffic
    ord('2'): dict(name="HIGHWAY",   soft_zone=55),   # was 30°
    ord('3'): dict(name="MOTORWAY",  soft_zone=40),   # was 20° — still precise but not hair-trigger
}
_speed_key = ord('1')

# Mouth
MOUTH_OPEN_THRESH   = 0.38
MOUTH_CLOSE_THRESH  = 0.28

# Horn — brow raise
BROW_RAISE_THRESH   = 0.65    # higher = needs more deliberate raise to avoid accidents
BROW_CONFIRM_FRAMES = 8       # must hold raise this many frames before horn fires

# Head tilt → indicator
HEAD_TILT_THRESH    = 0.05    # normalised eye-height difference to trigger (raise if too sensitive)
HEAD_TILT_RETURN    = 0.02    # must return within this of level before next toggle is allowed
HEAD_TILT_CONFIRM   = 6       # frames to hold tilt before toggling indicator
HEAD_TILT_COOLDOWN  = 1.5     # seconds between toggles

# Key bindings
KEY_HORN       = 'h'
KEY_IND_LEFT   = ','
KEY_IND_RIGHT  = '.'

# ─── COLORS ────────────────────────────────────────────────────────────────────
CLR_WHEEL   = (80, 200, 255)
CLR_LEFT    = (60, 120, 255)
CLR_RIGHT   = (50, 220, 140)
CLR_NEUTRAL = (200, 200, 200)
CLR_TEXT    = (255, 255, 255)
CLR_ACCENT  = (0,  180, 255)
CLR_HAND_L  = (255, 130,  60)
CLR_HAND_R  = ( 60, 230, 130)
CLR_GAS     = ( 50, 220, 140)
CLR_BRAKE   = ( 50,  80, 255)
CLR_HORN    = (  0, 220, 255)
CLR_GEAR    = (255, 220,  50)
CLR_IND     = ( 30, 180, 255)

mp_hands   = mp_hands_module
mp_drawing = mp_drawing_module
mp_face    = mp_face_mesh_module

# ── Face mesh landmark indices ─────────────────────────────────────────────────
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

# Head tilt — eye corner landmarks
# In selfie view (flipped): person's LEFT eye is on the LEFT side of screen
L_EYE_OUTER, L_EYE_INNER = 263, 362   # person's left eye corners
R_EYE_OUTER, R_EYE_INNER = 133,  33   # person's right eye corners

# Inner brow tops used for brow-raise detection
L_BROW_INNER, R_BROW_INNER = 105, 334
NOSE_TIP, CHIN = 1, 152


# ─── THREADED CAMERA ──────────────────────────────────────────────────────────
class CameraStream:
    """Background-thread capture — main loop always gets the freshest frame."""
    def __init__(self, src, backend):
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self._thread.start(); return self

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()

    def is_opened(self): return self.cap.isOpened()


# ─── CAMERA AUTO-DETECTION ────────────────────────────────────────────────────
def find_camera(backend, preferred=None):
    """
    Scan camera indices 0-4 and return (index, cv2.VideoCapture) for the first
    one that opens and delivers a real frame.  If preferred is set it is tried
    first so a manually-configured index still takes priority.
    Raises RuntimeError with actionable advice when nothing works.
    """
    candidates = list(range(5))
    if preferred is not None:
        candidates = [preferred] + [i for i in candidates if i != preferred]

    for idx in candidates:
        print(f"[CAM] Trying index {idx} ...", end=" ", flush=True)
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print("not found")
            cap.release()
            continue
        # warm-up: drain stale buffer frames
        for _ in range(3):
            cap.read()
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"OK  ({w}x{h})")
            return idx, cap
        print("no frame")
        cap.release()

    raise RuntimeError(
        "No working camera found on indices 0-4.\n"
        "  • Laptop built-in camera: make sure no other app (Teams, OBS,\n"
        "    Discord, Zoom) is using it — close those apps and retry.\n"
        "  • USB webcam: unplug and replug, then retry.\n"
        "  • Check Device Manager → Cameras for driver errors.\n"
        "  • If you have both built-in and USB, unplug the USB first to let\n"
        "    auto-detect grab the built-in, then replug after the script starts.")


# ─── GEOMETRY / GESTURE HELPERS ───────────────────────────────────────────────
def compute_head_tilt(lms):
    """
    Returns normalised vertical offset between eye centres.
      Positive  → head tilted RIGHT (right ear drops)
      Negative  → head tilted LEFT  (left ear drops)
    Uses the average Y of each eye's two corner landmarks.
    Works correctly because the frame is already flipped (selfie view) before
    processing, so person's left eye really is on the left side of the image.
    """
    ly = (lms[L_EYE_OUTER].y + lms[L_EYE_INNER].y) / 2.0  # person's left eye  avg Y
    ry = (lms[R_EYE_OUTER].y + lms[R_EYE_INNER].y) / 2.0  # person's right eye avg Y
    # head tilted right → right eye drops → ry > ly → positive result
    return ry - ly


def compute_mar(lms, w, h):
    top    = np.array([lms[MOUTH_TOP   ].x * w, lms[MOUTH_TOP   ].y * h])
    bottom = np.array([lms[MOUTH_BOTTOM].x * w, lms[MOUTH_BOTTOM].y * h])
    left   = np.array([lms[MOUTH_LEFT  ].x * w, lms[MOUTH_LEFT  ].y * h])
    right  = np.array([lms[MOUTH_RIGHT ].x * w, lms[MOUTH_RIGHT ].y * h])
    return np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6)


def compute_brow_raise(lms, w, h):
    """Normalised ratio — higher = brows more raised."""
    nose_y = lms[NOSE_TIP].y
    chin_y = lms[CHIN].y
    face_h = max(abs(chin_y - nose_y), 1e-4)
    brow_y = (lms[L_BROW_INNER].y + lms[R_BROW_INNER].y) / 2.0
    return (nose_y - brow_y) / face_h



# ─── STEERING MATH ────────────────────────────────────────────────────────────
def tilt_to_axis(angle_deg, soft_zone):
    """
    Maps tilt angle → joystick axis [-1, +1].
    Dead zone:  |angle| < DEAD_ZONE_DEG            →  0
    Power ramp: dead_zone … soft_zone               →  0 … ±1  (via STEERING_CURVE)
    Beyond soft_zone                                →  clamped ±1
    """
    if abs(angle_deg) < DEAD_ZONE_DEG:
        return 0.0
    sign = 1.0 if angle_deg > 0 else -1.0
    raw  = (abs(angle_deg) - DEAD_ZONE_DEG) / max(soft_zone - DEAD_ZONE_DEG, 1)
    return sign * min(raw, 1.0) ** STEERING_CURVE


# ─── CONTROLLER ───────────────────────────────────────────────────────────────
class SteeringController:
    def __init__(self):
        if not _GAMEPAD_AVAILABLE:
            raise RuntimeError(
                "vgamepad not available.\n"
                "  1. Install ViGEm Bus Driver: https://github.com/nefarius/ViGEmBus/releases\n"
                "  2. pip install vgamepad")
        self.pad = vg.VX360Gamepad()
        self.kb  = KeyboardController()

        # state
        self._steer      = 0.0
        self._ema_angle  = 0.0
        self._mouth_open = False
        self._key_up     = False
        self._key_dn     = False
        self._horn_held  = False
        self._ind_left   = False    # indicator on/off state
        self._ind_right  = False

        # horn debounce
        self._brow_frames = 0

        # head-tilt indicator debounce
        self._lwf = 0;  self._rwf = 0          # frame counters
        self._lwt = 0.0; self._rwt = 0.0       # cooldown timestamps
        self._lwfire = False; self._rwfire = False


    # ── internal ──────────────────────────────────────────────────────────────
    def _commit(self):
        try:
            self.pad.left_joystick_float(x_value_float=self._steer, y_value_float=0.0)
            self.pad.update()
        except Exception:
            pass   # ViGEm can hiccup briefly; ignore single-frame failures

    def _kpress(self, key):
        if key == Key.up   and not self._key_up: self.kb.press(Key.up);   self._key_up = True
        if key == Key.down and not self._key_dn: self.kb.press(Key.down); self._key_dn = True

    def _krelease(self, key):
        if key == Key.up   and self._key_up: self.kb.release(Key.up);   self._key_up = False
        if key == Key.down and self._key_dn: self.kb.release(Key.down); self._key_dn = False

    def _tap(self, char): self.kb.press(char); self.kb.release(char)

    # ── steering ──────────────────────────────────────────────────────────────
    def update_steering(self, left_wrist, right_wrist, soft_zone):
        dx  = right_wrist[0] - left_wrist[0]
        dy  = right_wrist[1] - left_wrist[1]
        raw = math.degrees(math.atan2(dy, dx))

        # EMA smoothing — much smoother than simple moving average
        self._ema_angle = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema_angle
        self._steer     = tilt_to_axis(self._ema_angle, soft_zone)
        self._commit()

        if   self._steer < -0.05: direction = "LEFT"
        elif self._steer >  0.05: direction = "RIGHT"
        else:                     direction = "STRAIGHT"
        return self._ema_angle, direction, abs(self._steer)

    def ease_to_center(self):
        """Gradually return axis to 0 instead of snapping when hands leave frame."""
        if abs(self._steer) < RECENTER_RATE:
            self._steer = 0.0
        else:
            self._steer -= math.copysign(RECENTER_RATE, self._steer)
        self._commit()

    def release_steering(self):
        self._ema_angle = 0.0
        self._steer     = 0.0
        self._commit()

    # ── throttle ──────────────────────────────────────────────────────────────
    def update_throttle(self, mar):
        if   mar > MOUTH_OPEN_THRESH:  self._mouth_open = True
        elif mar < MOUTH_CLOSE_THRESH: self._mouth_open = False
        if self._mouth_open: self._kpress(Key.down); self._krelease(Key.up)
        else:                self._kpress(Key.up);   self._krelease(Key.down)
        return self._mouth_open

    def release_throttle(self):
        self._krelease(Key.up); self._krelease(Key.down)

    # ── horn ──────────────────────────────────────────────────────────────────
    def update_horn(self, brow_raise):
        # Must hold the raise for BROW_CONFIRM_FRAMES consecutive frames
        # before the horn fires — prevents accidental triggers from head movement
        if brow_raise > BROW_RAISE_THRESH:
            self._brow_frames += 1
        else:
            self._brow_frames = 0

        should_horn = self._brow_frames >= BROW_CONFIRM_FRAMES
        if should_horn and not self._horn_held:
            self.kb.press(KEY_HORN);   self._horn_held = True
        elif not should_horn and self._horn_held:
            self.kb.release(KEY_HORN); self._horn_held = False

    def release_horn(self):
        if self._horn_held:
            self.kb.release(KEY_HORN); self._horn_held = False

    # ── indicators ────────────────────────────────────────────────────────────
    def update_indicators(self, head_tilt, now):
        """
        Tilt head LEFT  → right indicator  (reversed from raw value — corrected per user)
        Tilt head RIGHT → left  indicator
        Must hold the tilt for HEAD_TILT_CONFIRM frames, then cooldown resets.
        Must return close to level before the same side can fire again.
        """
        # ── left tilt → RIGHT indicator ───────────────────────────────────────
        if head_tilt < -HEAD_TILT_THRESH:
            self._lwf += 1
        else:
            if head_tilt > -HEAD_TILT_RETURN:
                self._lwf = 0; self._lwfire = False
        if self._lwf >= HEAD_TILT_CONFIRM and not self._lwfire and now - self._lwt > HEAD_TILT_COOLDOWN:
            self._ind_right = not self._ind_right
            self._lwfire    = True
            self._lwt       = now
            self._tap(KEY_IND_RIGHT)

        # ── right tilt → LEFT indicator ───────────────────────────────────────
        if head_tilt > HEAD_TILT_THRESH:
            self._rwf += 1
        else:
            if head_tilt < HEAD_TILT_RETURN:
                self._rwf = 0; self._rwfire = False
        if self._rwf >= HEAD_TILT_CONFIRM and not self._rwfire and now - self._rwt > HEAD_TILT_COOLDOWN:
            self._ind_left = not self._ind_left
            self._rwfire   = True
            self._rwt      = now
            self._tap(KEY_IND_LEFT)

    # ── release all ───────────────────────────────────────────────────────────
    def release_all(self):
        self.release_steering()
        self.release_throttle()
        self.release_horn()

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def steer_axis(self):  return self._steer
    @property
    def mouth_open(self):  return self._mouth_open
    @property
    def horn_held(self):   return self._horn_held
    @property
    def ind_left(self):    return self._ind_left
    @property
    def ind_right(self):   return self._ind_right


# ─── DRAW ─────────────────────────────────────────────────────────────────────
def draw_steering_wheel(frame, center, steer_axis, angle_deg):
    h, w   = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center
    color  = CLR_LEFT if steer_axis < -0.05 else (CLR_RIGHT if steer_axis > 0.05 else CLR_NEUTRAL)

    cv2.circle(frame, (cx+3, cy+3), radius, (0,0,0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)
    for sa in (0, 120, 240):
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.40 * math.cos(rad))
        y1 = int(cy - radius * 0.40 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1,y1), (x2,y2), color, 2)
    cv2.circle(frame, (cx, cy), 6, color, -1)
    if abs(steer_axis) > 0.05:
        arc = int(abs(steer_axis) * 90)
        sa  = -arc if steer_axis > 0 else 180
        ea  =  0   if steer_axis > 0 else 180 - arc
        cv2.ellipse(frame, (cx,cy), (radius,radius), 0, sa, ea, color, 5)


def draw_hud(frame, angle, direction, steer_axis, both_hands, face_visible,
             mouth_open, mar, fps, ctrl, profile_name, now):
    h, w = frame.shape[:2]
    f    = cv2.FONT_HERSHEY_SIMPLEX

    # dark panel
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 155), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # ── steering bar ─────────────────────────────────────────────────────────
    bw = int(w * 0.55); bh = 16; bx = (w - bw) // 2; by = h - 115
    mid = bx + bw // 2
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40, 40, 55), -1)
    cv2.rectangle(frame, (mid-2, by-4), (mid+2, by+bh+4), (180, 180, 180), -1)
    fill = int((bw // 2) * abs(steer_axis))
    if steer_axis < -0.01 and fill:
        cv2.rectangle(frame, (mid-fill, by), (mid, by+bh), CLR_LEFT, -1)
    elif steer_axis > 0.01 and fill:
        cv2.rectangle(frame, (mid, by), (mid+fill, by+bh), CLR_RIGHT, -1)

    dc = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, " <- LEFT",          (bx, by-10),            f, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->",          (bx+bw-80, by-10),      f, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, f"{steer_axis:+.2f}", (mid-30, by+bh+28),    f, 0.75, dc,        2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"tilt {angle:+.1f}",   (bx, h-68), f, 0.45, CLR_TEXT, 1)
    cv2.putText(frame, f"[{profile_name}]",         (bx+bw-120, h-68), f, 0.48, CLR_ACCENT, 1)

    # ── throttle ─────────────────────────────────────────────────────────────
    tc = CLR_BRAKE if mouth_open else CLR_GAS
    tl = "BRAKE" if mouth_open else "GAS"
    if face_visible:
        cv2.putText(frame, f"{tl} {mar:.2f}", (10, h-68), f, 0.47, tc, 1)
    else:
        cv2.putText(frame, "NO FACE",         (10, h-68), f, 0.47, (100,100,100), 1)

    # ── indicators ───────────────────────────────────────────────────────────
    lc = CLR_LEFT  if ctrl.ind_left  else (50, 50, 60)
    rc = CLR_RIGHT if ctrl.ind_right else (50, 50, 60)
    lw = 2 if ctrl.ind_left  else 1
    rw = 2 if ctrl.ind_right else 1
    cv2.putText(frame, "<<",   (10,  60), f, 0.9, lc, lw)
    cv2.putText(frame, ">>",   (w-60, 60), f, 0.9, rc, rw)

    # ── horn ─────────────────────────────────────────────────────────────────
    if ctrl.horn_held:
        cv2.putText(frame, "HORN!", (w//2-38, 60), f, 0.85, CLR_HORN, 2)

    # ── FPS / hand status ─────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS:{fps:.0f}", (w-75, 30), f, 0.5, CLR_ACCENT, 1)
    hs  = "HANDS OK" if both_hands else "SHOW BOTH HANDS"
    hc  = (60, 220, 60) if both_hands else (0, 80, 255)
    cv2.putText(frame, hs, (10, 30), f, 0.5, hc, 1)

    # ── mouth bar (right edge) ────────────────────────────────────────────────
    tx, ty0, ty1 = w-32, h-150, h-20
    cv2.rectangle(frame, (tx, ty0), (tx+18, ty1), (40,40,55), -1)
    if face_visible:
        fh = int((ty1-ty0) * min(mar / MOUTH_OPEN_THRESH, 1.0))
        cv2.rectangle(frame, (tx, ty1-fh), (tx+18, ty1), tc, -1)
    cv2.putText(frame, "M", (tx+3, ty0-5), f, 0.4, CLR_TEXT, 1)

    # ── speed profile legend ─────────────────────────────────────────────────
    cv2.putText(frame, "1=City 2=Hwy 3=Mway", (bx, h-20), f, 0.38, (120,120,120), 1)

    draw_steering_wheel(frame, (mid, h-32), steer_axis, angle)


def draw_hand_connection(frame, lw, rw):
    cv2.line(frame,   lw, rw, (30,100,200), 8)
    cv2.line(frame,   lw, rw, CLR_ACCENT, 2)
    cv2.circle(frame, lw, 10, CLR_HAND_L, -1); cv2.circle(frame, lw, 13, CLR_HAND_L, 2)
    cv2.circle(frame, rw, 10, CLR_HAND_R, -1); cv2.circle(frame, rw, 13, CLR_HAND_R, 2)
    cv2.circle(frame, ((lw[0]+rw[0])//2, (lw[1]+rw[1])//2), 7, CLR_WHEEL, -1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    global _speed_key

    if not _GAMEPAD_AVAILABLE:
        print("[ERROR] vgamepad or ViGEm Bus Driver not found.")
        print("  1. Install ViGEm Bus Driver: https://github.com/nefarius/ViGEmBus/releases")
        print("  2. Reboot, then: pip install vgamepad")
        input("Press Enter to exit...")
        return

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    try:
        cam_idx, _ = find_camera(backend, preferred=CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        input("Press Enter to exit...")
        return
    cam = CameraStream(cam_idx, backend).start()

    ctrl = SteeringController()
    tpe  = ThreadPoolExecutor(max_workers=2)   # parallel hand + face inference

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF)

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF)

    conn_style = mp_drawing.DrawingSpec(color=(80, 80, 100),    thickness=1)
    pt_style   = mp_drawing.DrawingSpec(color=(200, 200, 255),  thickness=1, circle_radius=2)

    prev_time   = time.perf_counter()
    angle       = 0.0
    direction   = "STRAIGHT"
    steer_axis  = 0.0
    mar         = 0.0
    lost_frames = 0
    face_visible = False

    print("=" * 60)
    print("  Virtual Steering Wheel v3  |  Q = quit")
    print("  1 = City  2 = Highway  3 = Motorway  (click app window)")
    print("  Open mouth=Brake | Closed=Gas")
    print("  Raise BOTH brows=Horn | Tilt head=Indicator")
    print("=" * 60)

    dead_frames = 0               # consecutive empty frames from camera
    MAX_DEAD    = 90              # ~3 s at 30 fps before reconnect attempt

    try:
        while True:
            ret, raw = cam.read()

            # ── camera watchdog: reconnect if feed dies ────────────────────────
            if not ret or raw is None:
                dead_frames += 1
                if dead_frames == MAX_DEAD:
                    print("[WARN] Camera feed lost — attempting reconnect...")
                    cam.stop()
                    try:
                        new_idx, _ = find_camera(backend, preferred=cam_idx)
                        cam_idx    = new_idx
                        cam        = CameraStream(cam_idx, backend).start()
                        dead_frames = 0
                        print("[INFO] Camera reconnected.")
                    except RuntimeError as e:
                        print(f"[ERROR] Reconnect failed: {e}")
                        break
                time.sleep(0.005)
                continue
            dead_frames = 0

            try:
                frame = raw.copy()
                if FLIP_CAMERA:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # downscale for inference — INTER_NEAREST is fastest for shrinking
                ph, pw = int(h * PROCESS_SCALE), int(w * PROCESS_SCALE)
                small  = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_NEAREST)
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                # ── parallel inference (hands + face run simultaneously) ────────
                fh = tpe.submit(hands.process,    rgb)
                ff = tpe.submit(face_mesh.process, rgb)
                try:
                    hr = fh.result(timeout=0.15)   # drop frame if inference hangs
                    fr = ff.result(timeout=0.15)
                except Exception:
                    hr = fr = type('R', (), {'multi_hand_landmarks': None,
                                             'multi_face_landmarks': None})()

                now     = time.perf_counter()
                profile = SPEED_PROFILES[_speed_key]

                # ── HANDS ─────────────────────────────────────────────────────
                # Assign left/right by screen X — never trust MediaPipe labels,
                # which can flip when hands are close together on a wheel.
                both_visible = False

                if hr.multi_hand_landmarks and len(hr.multi_hand_landmarks) >= 2:
                    detected = sorted(hr.multi_hand_landmarks,
                                      key=lambda lm: lm.landmark[0].x)
                    lm_l, lm_r = detected[0], detected[1]

                    for lm in (lm_l, lm_r):
                        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                                  pt_style, conn_style)

                    w0l = lm_l.landmark[0];  w0r = lm_r.landmark[0]
                    lx, ly   = w0l.x, w0l.y
                    rx, ry   = w0r.x, w0r.y
                    lpx, lpy = int(lx * w), int(ly * h)
                    rpx, rpy = int(rx * w), int(ry * h)

                    both_visible = True
                    lost_frames  = 0
                    draw_hand_connection(frame, (lpx, lpy), (rpx, rpy))
                    angle, direction, _ = ctrl.update_steering(
                        (lx, ly), (rx, ry), profile['soft_zone'])
                    steer_axis = ctrl.steer_axis
                else:
                    lost_frames += 1

                if not both_visible:
                    if lost_frames >= GRACE_FRAMES:
                        ctrl.ease_to_center()
                        steer_axis = ctrl.steer_axis
                        direction  = "STRAIGHT"

                # ── FACE ──────────────────────────────────────────────────────
                face_visible = False
                if fr.multi_face_landmarks:
                    lms          = fr.multi_face_landmarks[0].landmark
                    face_visible = True
                    mar       = compute_mar(lms, w, h)
                    brow      = compute_brow_raise(lms, w, h)
                    head_tilt = compute_head_tilt(lms)
                    ctrl.update_throttle(mar)
                    ctrl.update_horn(brow)
                    ctrl.update_indicators(head_tilt, now)
                else:
                    ctrl.release_throttle()
                    ctrl.release_horn()

                # ── FPS + RENDER ───────────────────────────────────────────────
                fps       = 1.0 / max(now - prev_time, 1e-9)
                prev_time = now

                draw_hud(frame, angle, direction, steer_axis,
                         both_visible, face_visible, ctrl.mouth_open, mar,
                         fps, ctrl, profile['name'], now)
                cv2.imshow("Virtual Steering Wheel v3", frame)

            except Exception as exc:
                # a single bad frame never crashes the app — log and continue
                print(f"[WARN] Frame skipped: {exc}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in SPEED_PROFILES:
                _speed_key = key

    finally:
        ctrl.release_all()
        tpe.shutdown(wait=False)
        hands.close()
        face_mesh.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All inputs released.")


if __name__ == "__main__":
    main()

"""
phone_server.py — Virtual Steering Wheel (Phone Edition)
=========================================================
Use any Android or iPhone as a wireless steering wheel — no webcam needed.

Controls
--------
  Tilt phone left/right  →  steering
  GAS / BRAKE buttons    →  throttle / brake
  SHIFT ▲ / SHIFT ▼     →  sequential gear up/down  (Xbox RB / LB)
  ◄ IND  IND ►           →  left / right indicator
  HORN                   →  horn (held)
  LIGHTS                 →  headlight toggle

Live ETS2 Gauges (speed, RPM, gear)
------------------------------------
  Uses the Funbit / RenCloud scs-sdk-plugin shared memory — no extra server needed.

  Setup (one time only):
    1. Download the plugin DLL from:
       https://github.com/RenCloud/scs-sdk-plugin/releases
    2. Copy  Win64/scs-telemetry.dll  into:
       Documents\Euro Truck Simulator 2\plugins\
       (create the plugins folder if it doesn't exist)
    3. Launch ETS2 — the plugin activates automatically.

  The dashboard shows LIVE speed/RPM/gear while ETS2 is running, and falls
  back to simulated data when the game is closed or the DLL isn't installed.

Setup
-----
1.  pip install websockets cryptography vgamepad pynput
2.  py phone_server.py
3.  Open the URL shown in the terminal on your phone (same Wi-Fi)
4.  Tap "Advanced → Proceed" for the SSL warning, then tap to start
"""

import asyncio, ctypes, json, math, mmap, os, socket, ssl, struct
import subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pynput.keyboard import Key, Controller as KeyboardController

try:
    import vgamepad as vg
    _GAMEPAD_AVAILABLE = True
except Exception:
    _GAMEPAD_AVAILABLE = False

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed.  Run:  pip install websockets")
    sys.exit(1)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HTTP_PORT  = 8443
WS_PORT    = 8765
CERT_FILE  = "phone_cert.pem"
KEY_FILE   = "phone_key.pem"

DEAD_ZONE_DEG  = 5      # degrees of tilt ignored around centre — increase if steering drifts
STEERING_CURVE = 2.5    # 1 = linear (harsh), 2.5 = gentler at small angles (recommended)
EMA_ALPHA      = 0.20   # smoothing: lower = smoother/laggier, higher = more responsive

# soft_zone = phone tilt angle (°) that produces full-lock steering.
# Larger = less sensitive.  Tune to taste.
SPEED_PROFILES = {
    1: dict(name="CITY",     soft_zone=70),   # was 40 — much gentler now
    2: dict(name="HIGHWAY",  soft_zone=55),   # was 30
    3: dict(name="MOTORWAY", soft_zone=40),   # was 20
}

KEY_HORN      = 'h'
KEY_IND_LEFT  = ','
KEY_IND_RIGHT = '.'
KEY_LIGHTS    = 'l'

# ─── ETS2 TELEMETRY — FUNBIT / SCS-SDK-PLUGIN SHARED MEMORY ───────────────────
#
# Reads live telemetry from the Funbit scs-sdk-plugin shared memory block.
# The plugin DLL writes truck data directly into Windows shared memory
# (no HTTP server needed).  Falls back to simulated data gracefully.
#
# Shared memory name: Local\SCSSdkTelemetry  (32 KB block)
# Struct offsets (scs-sdk-plugin v9/v10):
#   0x00  uint32  time          – game time ms
#   0x04  uint32  paused        – 1 = game paused
#   0x08  uint32  sdkActive     – 1 = plugin running & game active
#   0x0C  17×bool booleans      + 3 bytes padding  → total 20 bytes
#   0x20  float   speed         – m/s  (×3.6 → km/h)
#   0x24  float   engineRPM
#   ...   (many more floats for controls, coordinates, etc.)
#   0x10C int32   gear          – displayed gear (pos=forward, 0=N, neg=reverse)

_SHM_NAME       = "Local\\SCSSdkTelemetry"
_SHM_SIZE       = 32768
_OFF_SDK_ACTIVE = 0x08   # uint32
_OFF_SPEED      = 0x20   # float, m/s
_OFF_RPM        = 0x24   # float
_OFF_GEAR       = 0x10C  # int32, displayed gear


class ShmTelemetryReader:
    """
    Reads live ETS2 data from the Funbit scs-sdk-plugin Windows shared memory.
    Falls back to simulated data when the game isn't running or the plugin
    isn't installed.
    """

    def __init__(self):
        self._speed = 0.0
        self._rpm   = 800.0
        self._gear  = 0
        self._live  = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    # ── internal ──────────────────────────────────────────────────────────────
    def _open_shm(self):
        """Open the SCS shared memory block; return mmap or None."""
        try:
            FILE_MAP_READ = 0x0004
            handle = ctypes.windll.kernel32.OpenFileMappingW(
                FILE_MAP_READ, False, _SHM_NAME)
            if not handle:
                return None
            # Pass handle as the fileno; mmap accepts raw HANDLE on Windows
            return mmap.mmap(handle, _SHM_SIZE, access=mmap.ACCESS_READ)
        except Exception:
            return None

    def _read(self, shm):
        """Return (speed_kmh, rpm, gear) or None if SDK not active."""
        shm.seek(0)
        raw = shm.read(_SHM_SIZE)
        sdk_active = struct.unpack_from('<I', raw, _OFF_SDK_ACTIVE)[0]
        if not sdk_active:
            return None
        speed_ms = struct.unpack_from('<f', raw, _OFF_SPEED)[0]
        rpm      = struct.unpack_from('<f', raw, _OFF_RPM)[0]
        gear     = struct.unpack_from('<i', raw, _OFF_GEAR)[0]
        return speed_ms * 3.6, rpm, gear

    def _loop(self):
        shm = None
        while True:
            # ── try to open shared memory if not yet open ──────────────────
            if shm is None:
                shm = self._open_shm()
                if shm is None:
                    if self._live:
                        with self._lock:
                            self._live = False
                        print("[TELEM] SCS shared memory unavailable — using sim data.")
                    time.sleep(2.0)
                    continue
                print("[TELEM] Funbit SCS shared memory connected — live gauges active.")

            # ── read telemetry ─────────────────────────────────────────────
            try:
                result = self._read(shm)
                if result:
                    spd, rpm, gear = result
                    with self._lock:
                        self._speed = spd
                        self._rpm   = rpm
                        self._gear  = gear
                        self._live  = True
                else:
                    # SDK not active (game loading / main menu)
                    with self._lock:
                        self._live = False
            except Exception:
                # shared memory disappeared (game closed)
                try:
                    shm.close()
                except Exception:
                    pass
                shm = None
                with self._lock:
                    self._live = False
                print("[TELEM] SCS shared memory lost — will reconnect.")
                continue

            time.sleep(0.08)   # ~12 fps poll rate

    @property
    def snapshot(self):
        with self._lock:
            return {
                "speed": round(self._speed, 1),
                "rpm":   round(self._rpm),
                "gear":  self._gear,
                "live":  self._live,
            }


telemetry = ShmTelemetryReader()

# ─── STEERING MATH ─────────────────────────────────────────────────────────────
def tilt_to_axis(angle_deg, soft_zone):
    if abs(angle_deg) < DEAD_ZONE_DEG:
        return 0.0
    sign = 1.0 if angle_deg > 0 else -1.0
    raw  = (abs(angle_deg) - DEAD_ZONE_DEG) / max(soft_zone - DEAD_ZONE_DEG, 1)
    return sign * min(raw, 1.0) ** STEERING_CURVE


# ─── CONTROLLER ────────────────────────────────────────────────────────────────
class PhoneController:
    def __init__(self):
        self.kb = KeyboardController()
        self._ema        = 0.0
        self._steer      = 0.0
        self._key_up     = False
        self._key_dn     = False
        self._horn_held  = False
        self._prev_shift_up   = False
        self._prev_shift_down = False

        if _GAMEPAD_AVAILABLE:
            self.pad = vg.VX360Gamepad()
            print("[CTRL] Virtual Xbox gamepad ready.")
        else:
            self.pad = None
            print("[WARN] vgamepad not available — keyboard fallback only.")

    def update(self, tilt, gas, brake, horn, profile,
               shift_up, shift_down, ind_left, ind_right, lights):

        soft_zone = SPEED_PROFILES.get(profile, SPEED_PROFILES[1])['soft_zone']
        self._ema   = EMA_ALPHA * tilt + (1 - EMA_ALPHA) * self._ema
        self._steer = tilt_to_axis(self._ema, soft_zone)

        if self.pad:
            try:
                # Negate steer: phone tilt right → positive tilt → should turn right.
                # Xbox left-stick X: +1 = right in ETS2, so we negate to fix the
                # reversed-steering bug (tilting right was turning left in-game).
                self.pad.left_joystick_float(x_value_float=-self._steer, y_value_float=0.0)
                self.pad.update()
            except Exception:
                pass

        if gas:
            self._kpress(Key.up);   self._krelease(Key.down)
        elif brake:
            self._kpress(Key.down); self._krelease(Key.up)
        else:
            self._krelease(Key.up); self._krelease(Key.down)

        if horn and not self._horn_held:
            self.kb.press(KEY_HORN);   self._horn_held = True
        elif not horn and self._horn_held:
            self.kb.release(KEY_HORN); self._horn_held = False

        if shift_up and not self._prev_shift_up:
            self._tap_gamepad_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                                     if self.pad else None)
        if shift_down and not self._prev_shift_down:
            self._tap_gamepad_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
                                     if self.pad else None)
        self._prev_shift_up   = shift_up
        self._prev_shift_down = shift_down

        if ind_left:  self.kb.press(KEY_IND_LEFT);  self.kb.release(KEY_IND_LEFT)
        if ind_right: self.kb.press(KEY_IND_RIGHT); self.kb.release(KEY_IND_RIGHT)
        if lights:    self.kb.press(KEY_LIGHTS);    self.kb.release(KEY_LIGHTS)

    def _tap_gamepad_button(self, btn):
        if self.pad and btn is not None:
            try:
                self.pad.press_button(button=btn);  self.pad.update()
                time.sleep(0.05)
                self.pad.release_button(button=btn); self.pad.update()
            except Exception:
                pass

    def _kpress(self, key):
        if key == Key.up   and not self._key_up: self.kb.press(Key.up);   self._key_up = True
        if key == Key.down and not self._key_dn: self.kb.press(Key.down); self._key_dn = True

    def _krelease(self, key):
        if key == Key.up   and self._key_up: self.kb.release(Key.up);   self._key_up = False
        if key == Key.down and self._key_dn: self.kb.release(Key.down); self._key_dn = False

    def release_all(self):
        self._krelease(Key.up); self._krelease(Key.down)
        if self._horn_held: self.kb.release(KEY_HORN); self._horn_held = False
        if self.pad:
            try:
                self.pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.pad.update()
            except Exception:
                pass


ctrl = PhoneController()

# ─── CONNECTED CLIENTS (for telemetry push) ────────────────────────────────────
_clients: set = set()
_clients_lock = threading.Lock()


# ─── WEBSOCKET ─────────────────────────────────────────────────────────────────
async def ws_handler(websocket):
    addr = websocket.remote_address
    print(f"[WS]  Connected:    {addr[0]}")
    with _clients_lock:
        _clients.add(websocket)
    try:
        async for msg in websocket:
            try:
                d = json.loads(msg)
                ctrl.update(
                    tilt        = float(d.get('tilt',       0)),
                    gas         = bool (d.get('gas',        False)),
                    brake       = bool (d.get('brake',      False)),
                    horn        = bool (d.get('horn',       False)),
                    profile     = int  (d.get('profile',    1)),
                    shift_up    = bool (d.get('shiftUp',    False)),
                    shift_down  = bool (d.get('shiftDown',  False)),
                    ind_left    = bool (d.get('indLeft',    False)),
                    ind_right   = bool (d.get('indRight',   False)),
                    lights      = bool (d.get('lights',     False)),
                )
            except Exception as e:
                print(f"[WS]  Bad message: {e}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with _clients_lock:
            _clients.discard(websocket)
        ctrl.release_all()
        print(f"[WS]  Disconnected: {addr[0]}")


# ─── TELEMETRY PUSH TASK ────────────────────────────────────────────────────────
async def telemetry_push():
    """Push ETS2 telemetry (or sim data) to all connected phone clients ~10 fps."""
    while True:
        await asyncio.sleep(0.10)
        snap = telemetry.snapshot
        msg  = json.dumps({"t": snap})
        with _clients_lock:
            dead = set()
            for ws in list(_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            _clients.difference_update(dead)


# ─── CERTIFICATE ───────────────────────────────────────────────────────────────
def ensure_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    print("[CERT] Generating self-signed certificate (first run only)…")
    try:
        subprocess.run(["openssl","req","-x509","-newkey","rsa:2048",
                        "-keyout",KEY_FILE,"-out",CERT_FILE,"-days","730",
                        "-nodes","-subj","/CN=SteeringWheel"],
                       check=True, capture_output=True)
        print("[CERT] Done (openssl).")
        return
    except Exception:
        pass
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SteeringWheel")])
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.utcnow())
                .not_valid_after(datetime.datetime.utcnow()+datetime.timedelta(days=730))
                .sign(key, hashes.SHA256()))
        with open(KEY_FILE,"wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()))
        with open(CERT_FILE,"wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print("[CERT] Done (cryptography).")
    except ImportError:
        print("[ERROR] pip install cryptography  OR install openssl"); sys.exit(1)


# ─── HTML ──────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Wheel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;touch-action:manipulation}
html,body{width:100%;height:100%;overflow:hidden;background:#07070f;color:#fff;
          font-family:Arial,sans-serif;user-select:none}

/* ── rotate-to-landscape overlay ── */
#rot{position:fixed;inset:0;background:#07070f;z-index:200;
     display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px}
#rot span{font-size:52px;animation:spin 2s ease-in-out infinite}
#rot p{color:#555;font-size:14px;letter-spacing:1px}
@keyframes spin{0%,100%{transform:rotate(0deg)}50%{transform:rotate(90deg)}}
@media(orientation:landscape){#rot{display:none}}

/* ── tap-to-start overlay ── */
#tap{position:fixed;inset:0;background:#07070fdd;z-index:100;
     display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;cursor:pointer}
#tap h2{font-size:20px;letter-spacing:3px;color:#00c8ff}
#tap p{color:#444;font-size:12px;text-align:center;line-height:1.8}
#pulse{width:52px;height:52px;border-radius:50%;border:2px solid #00c8ff;
       animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(.85);opacity:.4}50%{transform:scale(1.1);opacity:1}}
#tap.hide{display:none}

/* ══ LANDSCAPE LAYOUT ══════════════════════════════════════════════════════ */
#app{display:none;width:100%;height:100%;flex-direction:column}
@media(orientation:landscape){#app{display:flex}}

/* ── top status strip ── */
#st{display:flex;align-items:center;justify-content:space-between;
    padding:4px 10px;background:#0e0e1c;flex-shrink:0;height:28px}
#dot{width:8px;height:8px;border-radius:50%;background:#333;
     margin-right:6px;display:inline-block;transition:background .3s}
#dot.ok{background:#00e676}#dot.err{background:#f44}
#lbl{font-size:11px;color:#666}
#tlive{font-size:10px;color:#555;margin-left:8px}
#tlive.on{color:#00e676}
#profs{display:flex;gap:4px}
.pb{padding:2px 8px;border:1px solid #222;border-radius:3px;background:none;
    color:#555;font-size:10px;cursor:pointer;transition:all .15s}
.pb.on{border-color:#00c8ff;color:#00c8ff;background:#091520}

/* ── body row ── */
#body{flex:1;display:flex;min-height:0;gap:5px;padding:5px}

#lcol{width:22%;display:flex;flex-direction:column;gap:5px}
#rcol{width:22%;display:flex;flex-direction:column;gap:5px}
#ccol{flex:1;display:flex;flex-direction:column;gap:4px;min-width:0}

.side-sm{flex-shrink:0;border:none;border-radius:8px;background:#0f0f20;
         color:#555;font-size:11px;font-weight:bold;letter-spacing:.5px;
         padding:0;height:28px;cursor:pointer;touch-action:none;transition:all .12s}
.side-sm.flash,#ind-l.flash,#ind-r.flash{color:#ffcc00;background:#1a1500;border:1px solid #ffcc00}

.shiftb{flex-shrink:0;border:none;border-radius:8px;background:#0f0f20;
        color:#777;font-size:13px;font-weight:bold;height:36px;
        cursor:pointer;touch-action:none;letter-spacing:1px;transition:filter .1s}
.shiftb.on{filter:brightness(2.2);color:#fff}

.bigb{flex:1;border:none;border-radius:12px;font-size:17px;font-weight:bold;
      letter-spacing:2px;cursor:pointer;display:flex;align-items:center;
      justify-content:center;touch-action:none;transition:filter .1s;min-height:0}
#bkb{background:linear-gradient(170deg,#4a0000,#991200);color:#ffaaaa}
#gsb{background:linear-gradient(170deg,#002800,#007515);color:#aaffaa}
.bigb.on{filter:brightness(1.6)}

/* ── dashboard canvas ── */
#dash{flex-shrink:0;width:100%;position:relative}
#cv{display:block;width:100%}

/* ── tilt bar ── */
#trow{flex-shrink:0;display:flex;align-items:center;gap:8px;padding:0 2px}
#tbg{flex:1;height:5px;background:#111120;border-radius:3px;position:relative}
#tfill{position:absolute;top:0;height:100%;border-radius:3px;transition:all .04s}
#tctr{position:absolute;left:50%;top:-3px;width:2px;height:11px;
      background:#333;transform:translateX(-50%)}
#tval{min-width:46px;text-align:right;font-size:11px;color:#00c8ff;font-weight:bold}

/* ── horn + lights row ── */
#hlrow{flex-shrink:0;display:grid;grid-template-columns:1fr 1fr;gap:5px}
.hlb{border:none;border-radius:8px;background:#0f0f20;color:#555;
     font-size:12px;font-weight:bold;padding:7px 4px;cursor:pointer;
     letter-spacing:.5px;touch-action:none;transition:all .12s}
#hornb2.on{color:#00c8ff;background:#001820}
#lightsb2.on{color:#ffe066;background:#1a1400}
.hlb.flash{color:#fff;background:#1a1a30}
</style>
</head>
<body>

<!-- rotate overlay -->
<div id="rot">
  <span>📱</span>
  <p>ROTATE TO LANDSCAPE</p>
</div>

<!-- tap-to-start -->
<div id="tap">
  <div id="pulse"></div>
  <h2>🎮 STEERING WHEEL</h2>
  <p>Tap anywhere to start<br>Hold phone in landscape and tilt left / right to steer</p>
</div>

<!-- main app -->
<div id="app">

  <div id="st">
    <div style="display:flex;align-items:center">
      <span id="dot"></span><span id="lbl">Waiting…</span>
      <span id="tlive">● SIM</span>
    </div>
    <div id="profs">
      <button class="pb on" data-p="1">CITY</button>
      <button class="pb"    data-p="2">HWY</button>
      <button class="pb"    data-p="3">MWAY</button>
    </div>
  </div>

  <div id="body">

    <!-- LEFT: IND ◄ | SHIFT ▼ | BRAKE -->
    <div id="lcol">
      <button class="side-sm" id="ind-l"
        ontouchstart="tapInd('l')" onmousedown="tapInd('l')">◄ IND</button>
      <button class="shiftb" id="sdnb"
        ontouchstart="tapShift('d')" onmousedown="tapShift('d')">▼ SHIFT</button>
      <button class="bigb" id="bkb"
        ontouchstart="sb2('b',true)"  ontouchend="sb2('b',false)"
        onmousedown="sb2('b',true)"   onmouseup="sb2('b',false)">BRAKE</button>
    </div>

    <!-- CENTER: dashboard | tilt | horn + lights -->
    <div id="ccol">
      <div id="dash"><canvas id="cv"></canvas></div>
      <div id="trow">
        <div id="tbg"><div id="tfill"></div><div id="tctr"></div></div>
        <div id="tval">0.0°</div>
      </div>
      <div id="hlrow">
        <button class="hlb" id="hornb2"
          ontouchstart="sh(true)"  ontouchend="sh(false)"
          onmousedown="sh(true)"   onmouseup="sh(false)">📯 HORN</button>
        <button class="hlb" id="lightsb2"
          ontouchstart="tapLights()" onmousedown="tapLights()">💡 LIGHTS</button>
      </div>
    </div>

    <!-- RIGHT: IND ► | SHIFT ▲ | GAS -->
    <div id="rcol">
      <button class="side-sm" id="ind-r"
        ontouchstart="tapInd('r')" onmousedown="tapInd('r')">IND ►</button>
      <button class="shiftb" id="supb"
        ontouchstart="tapShift('u')" onmousedown="tapShift('u')">▲ SHIFT</button>
      <button class="bigb" id="gsb"
        ontouchstart="sb2('g',true)"  ontouchend="sb2('g',false)"
        onmousedown="sb2('g',true)"   onmouseup="sb2('g',false)">GAS</button>
    </div>

  </div>
</div>

<script>
const WS_PORT = __WS_PORT__;

// ── state ────────────────────────────────────────────────────────────────────
let tilt=0, gas=false, brake=false, horn=false, profile=1;
let shiftUp=false, shiftDown=false;
let indLeft=false, indRight=false, lights=false;

// ── telemetry from server ─────────────────────────────────────────────────────
let tSpeed=0, tRpm=800, tGear=0, tLive=false;

// ── local gear tracking (used when NOT live) ─────────────────────────────────
let localGear=0, _pu=false, _pd=false;
function trackLocalGear(){
  if(shiftUp&&!_pu)  localGear=Math.min(12,localGear+1);
  if(shiftDown&&!_pd)localGear=Math.max(0,localGear-1);
  _pu=shiftUp; _pd=shiftDown;
}

// ── local sim fallback (used when NOT live) ───────────────────────────────────
const IDLE=680, MAX_RPM=2300, RED_RPM=1950;
const G_TOP=[0,14,26,40,56,72,90,108,120,132,143,153,160];
let simSpd=0, simRpm=IDLE;
function simTick(){
  const g=localGear;
  if(g===0){
    simSpd=Math.max(0,simSpd-1.2);
    simRpm=gas?Math.min(1600,simRpm+40):Math.max(IDLE,simRpm-25);
  }else{
    const top=G_TOP[Math.min(g,12)],prev=g>1?G_TOP[g-1]:0;
    if(gas)        simSpd=Math.min(simSpd+(g<3?1.8:.7),top);
    else if(brake) simSpd=Math.max(0,simSpd-3.5);
    else           simSpd=Math.max(0,simSpd-.25);
    const frac=Math.max(0,(simSpd-prev)/Math.max(top-prev,1));
    const base=IDLE+frac*(MAX_RPM-IDLE);
    simRpm+=(gas?Math.min(base*1.25,MAX_RPM):base-simRpm)*0.12;
    simRpm=Math.max(IDLE,Math.min(MAX_RPM,simRpm));
  }
}

// ── canvas gauges ─────────────────────────────────────────────────────────────
const cv=document.getElementById('cv');
const cx2=cv.getContext('2d');
let CW=0,CH=0,GR=0;

function sizeCanvas(){
  const el=document.getElementById('ccol');
  CW=el.clientWidth||1;
  const bodyH=document.getElementById('body').clientHeight||1;
  CH=Math.max(60,bodyH-72);
  cv.width=CW; cv.height=CH;
  GR=Math.min(CH*0.46,CW*0.19);
}

function gauge(cx,cy,r,mn,mx,val,unit,col,redAt){
  const SA=.75*Math.PI,EA=2.25*Math.PI,ARC=1.5*Math.PI;
  const pct=Math.max(0,Math.min(1,(val-mn)/(mx-mn)));
  const va=SA+pct*ARC, tr=r*.80, lw=Math.max(3,r*.13);
  cx2.beginPath();cx2.arc(cx,cy,tr,SA,EA,false);
  cx2.strokeStyle='#15152a';cx2.lineWidth=lw;cx2.lineCap='round';cx2.stroke();
  if(redAt!==undefined){
    const rp=(redAt-mn)/(mx-mn);
    cx2.beginPath();cx2.arc(cx,cy,tr,SA+rp*ARC,EA,false);
    cx2.strokeStyle='#280808';cx2.lineWidth=lw;cx2.stroke();
  }
  cx2.beginPath();cx2.arc(cx,cy,tr,SA,va,false);
  cx2.strokeStyle=col;cx2.lineWidth=lw*.75;cx2.lineCap='round';cx2.stroke();
  const ex=cx+tr*Math.cos(va),ey=cy+tr*Math.sin(va);
  cx2.beginPath();cx2.arc(ex,ey,lw*.5,0,Math.PI*2);cx2.fillStyle=col;cx2.fill();
  cx2.textAlign='center';
  cx2.fillStyle='#e8e8e8';cx2.font=`bold ${r*.33}px Arial`;
  cx2.fillText(Math.round(val),cx,cy+r*.1);
  cx2.fillStyle='#3a3a3a';cx2.font=`${r*.17}px Arial`;
  cx2.fillText(unit,cx,cy+r*.32);
}

function drawDash(){
  const spd  = tLive ? tSpeed : simSpd;
  const rpm  = tLive ? tRpm   : simRpm;
  const gear = tLive ? tGear  : localGear;

  cx2.clearRect(0,0,CW,CH);
  cx2.fillStyle='#07070f';cx2.fillRect(0,0,CW,CH);
  const cy=CH*.60, pad=GR*.15;
  const lcx=GR+pad, rcx=CW-GR-pad, gcx=CW/2;
  gauge(lcx,cy,GR,0,160,spd,'km/h','#00c8ff',130);
  gauge(rcx,cy,GR,0,2500,rpm,'RPM','#ff4060',RED_RPM);
  cx2.textAlign='center';
  cx2.fillStyle=gear===0?'#444':'#fff';
  cx2.font=`bold ${GR*.75}px Arial`;
  cx2.fillText(gear===0?'N':String(gear),gcx,cy+GR*.05);
  cx2.fillStyle='#2a2a2a';cx2.font=`${GR*.17}px Arial`;
  cx2.fillText('GEAR',gcx,cy+GR*.3);
  cx2.strokeStyle='#181828';cx2.lineWidth=1;
  [lcx+GR*.92,rcx-GR*.92].forEach(x=>{
    cx2.beginPath();cx2.moveTo(x,CH*.1);cx2.lineTo(x,CH*.9);cx2.stroke();
  });
}

// ── tilt bar ──────────────────────────────────────────────────────────────────
const tfill=document.getElementById('tfill'),tvalEl=document.getElementById('tval');
function updateTilt(){
  const abs=Math.abs(tilt);
  tvalEl.textContent=(tilt>=0?'+':'')+tilt.toFixed(1)+'°';
  if(tilt<-1)      tfill.style.cssText=`left:${50-abs/90*50}%;width:${abs/90*50}%;background:#4488ff`;
  else if(tilt>1)  tfill.style.cssText=`left:50%;width:${abs/90*50}%;background:#00e676`;
  else             tfill.style.cssText='width:0';
}

// ── button handlers ───────────────────────────────────────────────────────────
function sb2(t,s){
  if(t==='g'){gas=s;  document.getElementById('gsb').classList.toggle('on',s)}
  if(t==='b'){brake=s;document.getElementById('bkb').classList.toggle('on',s)}
}
function sh(s){horn=s;document.getElementById('hornb2').classList.toggle('on',s)}
function tapShift(d){
  if(d==='u'){shiftUp=true; flash('supb')}
  if(d==='d'){shiftDown=true;flash('sdnb')}
  // Send immediately so even the shortest tap registers
  if(ws&&ws.readyState===1){
    ws.send(JSON.stringify({tilt,gas,brake,horn,profile,shiftUp,shiftDown,indLeft,indRight,lights}));
    shiftUp=false;shiftDown=false;
  }
}
function tapInd(side){
  if(side==='l'){indLeft=true; flash('ind-l')}
  else          {indRight=true;flash('ind-r')}
}
function tapLights(){lights=true;flash('lightsb2')}
function flash(id){
  const el=document.getElementById(id);
  el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),220);
}

// ── profiles ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.pb').forEach(b=>b.addEventListener('click',()=>{
  profile=+b.dataset.p;
  document.querySelectorAll('.pb').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
}));

// ── WebSocket ─────────────────────────────────────────────────────────────────
const dot=document.getElementById('dot'),lbl=document.getElementById('lbl');
const tliveEl=document.getElementById('tlive');
let ws, sendLoop;

function connect(){
  ws=new WebSocket('wss://'+window.location.hostname+':'+WS_PORT);
  ws.onopen=()=>{
    dot.className='ok'; lbl.textContent='Connected ✓';
    sendLoop=setInterval(sendCtrl,33);
  };
  ws.onclose=ws.onerror=()=>{
    dot.className='err'; lbl.textContent='Reconnecting…';
    clearInterval(sendLoop); setTimeout(connect,2000);
  };
  ws.onmessage=e=>{
    try{
      const d=JSON.parse(e.data);
      if(d.t){
        tSpeed=d.t.speed||0; tRpm=d.t.rpm||800;
        tGear=d.t.gear||0;   tLive=!!d.t.live;
        tliveEl.textContent=tLive?'● LIVE':'● SIM';
        tliveEl.className=tLive?'on':'';
      }
    }catch(err){}
  };
}

function sendCtrl(){
  if(ws&&ws.readyState===1)
    ws.send(JSON.stringify({tilt,gas,brake,horn,profile,
      shiftUp,shiftDown,indLeft,indRight,lights}));
  indLeft=indRight=lights=false;
  shiftUp=shiftDown=false;
}

// ── orientation → tilt  ───────────────────────────────────────────────────────
// DeviceOrientationEvent axes are defined in the device's *natural* (portrait)
// frame.  When the phone is held in landscape the left/right lean of the screen
// no longer maps to gamma — it maps to beta (with a sign flip depending on
// which way the phone was rotated).
//
//   screen.orientation.angle:
//     0   = portrait
//     90  = landscape, top of phone pointing RIGHT  → use  -beta
//     270 = landscape, top of phone pointing LEFT   → use  +beta
//
// window.orientation (older iOS fallback):
//    -90  = same as screen angle 90
//     90  = same as screen angle 270

function getLandscapeAngle(){
  if(screen.orientation && screen.orientation.angle != null)
    return screen.orientation.angle;
  // iOS legacy
  const wo = window.orientation || 0;
  if(wo === -90) return 90;
  if(wo ===  90) return 270;
  return 0;
}

function onOri(e){
  const angle = getLandscapeAngle();
  let raw;
  if(angle === 90)                    raw = -(e.beta  || 0);   // top→right
  else if(angle === 270)              raw =  (e.beta  || 0);   // top→left
  else                                raw =  (e.gamma || 0);   // portrait
  tilt = Math.max(-90, Math.min(90, raw));
}

// ── animation loop ────────────────────────────────────────────────────────────
function frame(){
  trackLocalGear();
  if(!tLive) simTick();
  updateTilt();
  drawDash();
  requestAnimationFrame(frame);
}

function onResize(){ sizeCanvas(); drawDash(); }
window.addEventListener('resize', onResize);
window.addEventListener('orientationchange', ()=>setTimeout(onResize,150));

// ── start ─────────────────────────────────────────────────────────────────────
const tapEl=document.getElementById('tap');
async function start(){
  if(typeof DeviceOrientationEvent!=='undefined'&&
     typeof DeviceOrientationEvent.requestPermission==='function'){
    try{
      const r=await DeviceOrientationEvent.requestPermission();
      if(r!=='granted'){alert('Sensor access denied. Enable Motion in Safari Settings.');return;}
    }catch(e){}
  }
  window.addEventListener('deviceorientation',onOri,{passive:true});
  tapEl.classList.add('hide');
  connect();
  sizeCanvas();
  frame();
}
tapEl.addEventListener('click',start);
tapEl.addEventListener('touchend',e=>{e.preventDefault();start()});
</script>
</body>
</html>
"""


# ─── HTTP SERVER ───────────────────────────────────────────────────────────────
def _html():
    return _HTML.replace("__WS_PORT__", str(WS_PORT))

class _UI(BaseHTTPRequestHandler):
    def do_GET(self):
        b = _html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *_): pass

def _http(ssl_ctx):
    s = HTTPServer(("0.0.0.0", HTTP_PORT), _UI)
    s.socket = ssl_ctx.wrap_socket(s.socket, server_side=True)
    s.serve_forever()


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def _ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception: return "127.0.0.1"

async def main():
    ensure_cert()
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)
    threading.Thread(target=_http, args=(ssl_ctx,), daemon=True).start()
    ip = _ip()
    print(); print("=" * 60)
    print("  Virtual Steering Wheel — Phone Mode")
    print(f"\n  Open on your phone:  https://{ip}:{HTTP_PORT}")
    print("\n  ⚠  Tap  Advanced → Proceed  to skip the SSL warning.")
    print("     (Self-signed cert — safe on your LAN)")
    print("\n  ─ LIVE GAUGES ─────────────────────────────────────────")
    print("  Install the Funbit scs-sdk-plugin DLL once and gauges")
    print("  go live automatically whenever ETS2 is running.")
    print("  Download: https://github.com/RenCloud/scs-sdk-plugin/releases")
    print("  Place Win64/scs-telemetry.dll in:")
    print("    Documents\\Euro Truck Simulator 2\\plugins\\")
    print("  Without the DLL, gauges show simulated data.")
    print("\n  Ctrl+C to stop."); print("=" * 60); print()

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT, ssl=ssl_ctx):
        await telemetry_push()   # runs forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ctrl.release_all()
        print("\n[INFO] Stopped. All inputs released.")

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
    virtual_steering_wheel.py

echo.
echo ============================================================
echo  Done!
echo  Your app folder is:  dist\virtual_steering_wheel\
echo  Run it with:         dist\virtual_steering_wheel\steering_wheel.exe
echo.
echo  IMPORTANT: Also install ViGEm Bus Driver if not yet done:
echo  https://github.com/nefarius/ViGEmBus/releases
echo ============================================================
pause

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
| `1` | CITY | 70° tilt | City streets, tight turns |
| `2` | HIGHWAY | 55° tilt | Open roads |
| `3` | MOTORWAY | 40° tilt | Highways, keeping a trailer stable |

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
| `DEAD_ZONE_DEG` | `5` | Degrees of tilt ignored at centre — increase if steering drifts |
| `STEERING_CURVE` | `2.5` | `1` = linear (twitchy), `2.5` = gentler at small angles (recommended) |
| `EMA_ALPHA` | `0.20` | Smoothing: lower = smoother but more lag; higher = more responsive |
| `BROW_RAISE_THRESH` | `0.65` | How extreme a brow raise is needed for horn — raise if accidental |
| `HEAD_TILT_THRESH` | `0.05` | How much head tilt triggers indicator — raise if accidental |
| `FLIP_CAMERA` | `True` | Mirror the feed (selfie mode). Try `False` if steering feels reversed |

For **phone_server.py**, the same `DEAD_ZONE_DEG`, `STEERING_CURVE`, and `EMA_ALPHA` settings appear at the top of that file too.

---

## 🎮 Works With Other Games Too

Any game that supports an **Xbox controller** or **keyboard arrows** will work:

- **Keyboard arrows**: Mouth open/closed controls gas/brake automatically
- **Xbox controller axis**: Steering uses analog left-stick X — works in any game that supports a controller

Games tested:
- Euro Truck Simulator 2 ✅
- American Truck Simulator ✅
- Any browser racing game using arrow keys ✅

"""
Virtual Steering Wheel — All-in-One App
========================================
Camera mode  : webcam + MediaPipe hand/face tracking → steering / gas / horn / indicators
Phone mode   : hold phone in landscape, tilt to steer; on-screen buttons for everything

Steering direction is identical in both modes:
  tilt/tilt-hands RIGHT  →  turn RIGHT in ETS2
  tilt/tilt-hands LEFT   →  turn LEFT  in ETS2

Live ETS2 gauges via Funbit scs-sdk-plugin shared memory — no extra server needed.
  Get DLL : https://github.com/RenCloud/scs-sdk-plugin/releases
  Copy     : Win64/scs-telemetry.dll → Documents//Euro Truck Simulator 2//plugins//

Requirements
  pip install mediapipe==0.10.14 opencv-python vgamepad pynput numpy
              websockets cryptography Pillow qrcode[pil]
  Windows  : ViGEm Bus Driver  https://github.com/nefarius/ViGEmBus/releases

Run
  py -3.11 virtual_steering_wheel.py
"""

from __future__ import annotations
import asyncio, ctypes, json, math, mmap, os, platform, queue
import socket, ssl, struct, subprocess, sys, threading, time
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

KEY_HORN      = 'h'
KEY_IND_LEFT  = ','
KEY_IND_RIGHT = '.'
KEY_LIGHTS    = 'l'


# ════════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY TELEMETRY  (Funbit / RenCloud scs-sdk-plugin)
# ════════════════════════════════════════════════════════════════════════════════

_SHM_NAME       = "Local\\SCSSdkTelemetry"
_SHM_SIZE       = 32768
_OFF_SDK_ACTIVE = 0x08    # uint32 — 1 when plugin+game are active
_OFF_SPEED      = 0x20    # float  — truck speed in m/s  (×3.6 = km/h)
_OFF_RPM        = 0x24    # float  — engine RPM
_OFF_GEAR       = 0x10C   # int32  — displayed gear (positive=forward, 0=N, negative=reverse)


class ShmTelemetryReader:
    """
    Reads live ETS2 data from the Funbit scs-sdk-plugin Windows shared memory block.
    Falls back to simulated data gracefully when the game is not running or the
    plugin DLL is not installed.
    """
    def __init__(self):
        self._speed = 0.0
        self._rpm   = 800.0
        self._gear  = 0
        self._live  = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open_shm(self):
        try:
            handle = ctypes.windll.kernel32.OpenFileMappingW(0x0004, False, _SHM_NAME)
            if not handle:
                return None
            return mmap.mmap(handle, _SHM_SIZE, access=mmap.ACCESS_READ)
        except Exception:
            return None

    def _read(self, shm):
        shm.seek(0)
        raw = shm.read(_SHM_SIZE)
        if not struct.unpack_from('<I', raw, _OFF_SDK_ACTIVE)[0]:
            return None
        return (
            struct.unpack_from('<f', raw, _OFF_SPEED)[0] * 3.6,
            struct.unpack_from('<f', raw, _OFF_RPM)[0],
            struct.unpack_from('<i', raw, _OFF_GEAR)[0],
        )

    def _loop(self):
        shm = None
        while True:
            if shm is None:
                shm = self._open_shm()
                if shm is None:
                    if self._live:
                        with self._lock: self._live = False
                        print('[TELEM] SCS shared memory unavailable — using simulated data.')
                    time.sleep(2.0)
                    continue
                print('[TELEM] Funbit SCS shared memory connected — live gauges active.')
            try:
                result = self._read(shm)
                if result:
                    spd, rpm, gear = result
                    with self._lock:
                        self._speed = spd; self._rpm = rpm
                        self._gear  = gear; self._live = True
                else:
                    with self._lock: self._live = False
            except Exception:
                try: shm.close()
                except Exception: pass
                shm = None
                with self._lock: self._live = False
                print('[TELEM] SCS shared memory lost — will reconnect on next poll.')
                continue
            time.sleep(0.08)

    @property
    def snapshot(self) -> dict:
        with self._lock:
            return {'speed': round(self._speed, 1),
                    'rpm':   round(self._rpm),
                    'gear':  self._gear,
                    'live':  self._live}


_telemetry = ShmTelemetryReader()   # singleton shared by both modes


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

        if _GAMEPAD:
            self.pad = vg.VX360Gamepad()
            print('[PHONE] Virtual Xbox gamepad ready.')
        else:
            self.pad = None
            print('[PHONE] vgamepad not available — keyboard fallback only.')

    def update(self, tilt, gas, brake, horn, profile,
               shift_up, shift_down, ind_left, ind_right, lights):
        soft_zone   = _PHONE_PROFILES.get(profile, _PHONE_PROFILES[1])['soft_zone']
        self._ema   = EMA_ALPHA * tilt + (1 - EMA_ALPHA) * self._ema
        self._steer = tilt_to_axis(self._ema, soft_zone)

        if self.pad:
            try:
                # Phone mode negates the steer axis so that:
                #   "tilt phone right" → same right-turn direction as "tilt hands right"
                # The DeviceOrientation positive-tilt sign is opposite to the
                # atan2 camera geometry sign, so negation aligns both modes.
                self.pad.left_joystick_float(x_value_float=-self._steer, y_value_float=0.0)
                self.pad.update()
            except Exception:
                pass

        if self.kb:
            if gas:   self._kp(_Key.up);   self._kr(_Key.down)
            elif brake: self._kp(_Key.down); self._kr(_Key.up)
            else:     self._kr(_Key.up);   self._kr(_Key.down)

            if horn and not self._horn_held:
                self.kb.press(KEY_HORN);   self._horn_held = True
            elif not horn and self._horn_held:
                self.kb.release(KEY_HORN); self._horn_held = False

            if shift_up   and not self._prev_su: self._shift(True)
            if shift_down and not self._prev_sd: self._shift(False)
            self._prev_su = shift_up; self._prev_sd = shift_down

            if ind_left:  self.kb.press(KEY_IND_LEFT);  self.kb.release(KEY_IND_LEFT)
            if ind_right: self.kb.press(KEY_IND_RIGHT); self.kb.release(KEY_IND_RIGHT)
            if lights:    self.kb.press(KEY_LIGHTS);    self.kb.release(KEY_LIGHTS)

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
#rot{position:fixed;inset:0;background:#07070f;z-index:200;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px}
#rot span{font-size:52px;animation:spin 2s ease-in-out infinite}
#rot p{color:#555;font-size:14px;letter-spacing:1px}
@keyframes spin{0%,100%{transform:rotate(0deg)}50%{transform:rotate(90deg)}}
@media(orientation:landscape){#rot{display:none}}
#tap{position:fixed;inset:0;background:#07070fdd;z-index:100;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;cursor:pointer}
#tap h2{font-size:20px;letter-spacing:3px;color:#00c8ff}
#tap p{color:#444;font-size:12px;text-align:center;line-height:1.8}
#pulse{width:52px;height:52px;border-radius:50%;border:2px solid #00c8ff;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(.85);opacity:.4}50%{transform:scale(1.1);opacity:1}}
#tap.hide{display:none}
#app{display:none;width:100%;height:100%;flex-direction:column}
@media(orientation:landscape){#app{display:flex}}
#st{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:#0e0e1c;flex-shrink:0;height:28px}
#dot{width:8px;height:8px;border-radius:50%;background:#333;margin-right:6px;display:inline-block;transition:background .3s}
#dot.ok{background:#00e676}#dot.err{background:#f44}
#lbl{font-size:11px;color:#666}
#tlive{font-size:10px;color:#555;margin-left:8px}
#tlive.on{color:#00e676}
#profs{display:flex;gap:4px}
.pb{padding:2px 8px;border:1px solid #222;border-radius:3px;background:none;color:#555;font-size:10px;cursor:pointer;transition:all .15s}
.pb.on{border-color:#00c8ff;color:#00c8ff;background:#091520}
#body{flex:1;display:flex;min-height:0;gap:5px;padding:5px}
#lcol{width:22%;display:flex;flex-direction:column;gap:5px}
#rcol{width:22%;display:flex;flex-direction:column;gap:5px}
#ccol{flex:1;display:flex;flex-direction:column;gap:4px;min-width:0}
.side-sm{flex-shrink:0;border:none;border-radius:8px;background:#0f0f20;color:#555;font-size:11px;font-weight:bold;letter-spacing:.5px;padding:0;height:28px;cursor:pointer;touch-action:none;transition:all .12s}
.side-sm.flash,#ind-l.flash,#ind-r.flash{color:#ffcc00;background:#1a1500;border:1px solid #ffcc00}
.shiftb{flex-shrink:0;border:none;border-radius:8px;background:#0f0f20;color:#777;font-size:13px;font-weight:bold;height:36px;cursor:pointer;touch-action:none;letter-spacing:1px;transition:filter .1s}
.shiftb.on{filter:brightness(2.2);color:#fff}
.bigb{flex:1;border:none;border-radius:12px;font-size:17px;font-weight:bold;letter-spacing:2px;cursor:pointer;display:flex;align-items:center;justify-content:center;touch-action:none;transition:filter .1s;min-height:0}
#bkb{background:linear-gradient(170deg,#4a0000,#991200);color:#ffaaaa}
#gsb{background:linear-gradient(170deg,#002800,#007515);color:#aaffaa}
.bigb.on{filter:brightness(1.6)}
#dash{flex-shrink:0;width:100%;position:relative}
#cv{display:block;width:100%}
#trow{flex-shrink:0;display:flex;align-items:center;gap:8px;padding:0 2px}
#tbg{flex:1;height:5px;background:#111120;border-radius:3px;position:relative}
#tfill{position:absolute;top:0;height:100%;border-radius:3px;transition:all .04s}
#tctr{position:absolute;left:50%;top:-3px;width:2px;height:11px;background:#333;transform:translateX(-50%)}
#tval{min-width:46px;text-align:right;font-size:11px;color:#00c8ff;font-weight:bold}
#hlrow{flex-shrink:0;display:grid;grid-template-columns:1fr 1fr;gap:5px}
.hlb{border:none;border-radius:8px;background:#0f0f20;color:#555;font-size:12px;font-weight:bold;padding:7px 4px;cursor:pointer;letter-spacing:.5px;touch-action:none;transition:all .12s}
#hornb2.on{color:#00c8ff;background:#001820}
#lightsb2.on{color:#ffe066;background:#1a1400}
.hlb.flash{color:#fff;background:#1a1a30}
</style>
</head>
<body>
<div id="rot"><span>📱</span><p>ROTATE TO LANDSCAPE</p></div>
<div id="tap">
  <div id="pulse"></div>
  <h2>🎮 STEERING WHEEL</h2>
  <p>Tap anywhere to start<br>Hold phone in landscape and tilt left / right to steer</p>
</div>
<div id="app">
  <div id="st">
    <div style="display:flex;align-items:center">
      <span id="dot"></span><span id="lbl">Waiting…</span>
      <span id="tlive">● SIM</span>
    </div>
    <div id="profs">
      <button class="pb on" data-p="1">CITY</button>
      <button class="pb"    data-p="2">HWY</button>
      <button class="pb"    data-p="3">MWAY</button>
    </div>
  </div>
  <div id="body">
    <div id="lcol">
      <button class="side-sm" id="ind-l" ontouchstart="tapInd('l')" onmousedown="tapInd('l')">◄ IND</button>
      <button class="shiftb" id="sdnb"
        ontouchstart="tapShift('d')" onmousedown="tapShift('d')">▼ SHIFT</button>
      <button class="bigb" id="bkb"
        ontouchstart="sb2('b',true)" ontouchend="sb2('b',false)"
        onmousedown="sb2('b',true)" onmouseup="sb2('b',false)">BRAKE</button>
    </div>
    <div id="ccol">
      <div id="dash"><canvas id="cv"></canvas></div>
      <div id="trow">
        <div id="tbg"><div id="tfill"></div><div id="tctr"></div></div>
        <div id="tval">0.0°</div>
      </div>
      <div id="hlrow">
        <button class="hlb" id="hornb2"
          ontouchstart="sh(true)" ontouchend="sh(false)"
          onmousedown="sh(true)" onmouseup="sh(false)">📯 HORN</button>
        <button class="hlb" id="lightsb2"
          ontouchstart="tapLights()" onmousedown="tapLights()">💡 LIGHTS</button>
      </div>
    </div>
    <div id="rcol">
      <button class="side-sm" id="ind-r" ontouchstart="tapInd('r')" onmousedown="tapInd('r')">IND ►</button>
      <button class="shiftb" id="supb"
        ontouchstart="tapShift('u')" onmousedown="tapShift('u')">▲ SHIFT</button>
      <button class="bigb" id="gsb"
        ontouchstart="sb2('g',true)" ontouchend="sb2('g',false)"
        onmousedown="sb2('g',true)" onmouseup="sb2('g',false)">GAS</button>
    </div>
  </div>
</div>
<script>
const WS_PORT=__WS_PORT__;
let tilt=0,gas=false,brake=false,horn=false,profile=1;
let shiftUp=false,shiftDown=false,indLeft=false,indRight=false,lights=false;
let tSpeed=0,tRpm=800,tGear=0,tLive=false;
let localGear=0,_pu=false,_pd=false;
function trackLocalGear(){
  if(shiftUp&&!_pu)  localGear=Math.min(12,localGear+1);
  if(shiftDown&&!_pd)localGear=Math.max(0,localGear-1);
  _pu=shiftUp;_pd=shiftDown;
}
const IDLE=680,MAX_RPM=2300,RED_RPM=1950;
const G_TOP=[0,14,26,40,56,72,90,108,120,132,143,153,160];
let simSpd=0,simRpm=IDLE;
function simTick(){
  const g=localGear;
  if(g===0){simSpd=Math.max(0,simSpd-1.2);simRpm=gas?Math.min(1600,simRpm+40):Math.max(IDLE,simRpm-25);}
  else{
    const top=G_TOP[Math.min(g,12)],prev=g>1?G_TOP[g-1]:0;
    if(gas)simSpd=Math.min(simSpd+(g<3?1.8:.7),top);
    else if(brake)simSpd=Math.max(0,simSpd-3.5);
    else simSpd=Math.max(0,simSpd-.25);
    const frac=Math.max(0,(simSpd-prev)/Math.max(top-prev,1));
    const base=IDLE+frac*(MAX_RPM-IDLE);
    simRpm+=(gas?Math.min(base*1.25,MAX_RPM):base-simRpm)*0.12;
    simRpm=Math.max(IDLE,Math.min(MAX_RPM,simRpm));
  }
}
const cv=document.getElementById('cv'),cx2=cv.getContext('2d');
let CW=0,CH=0,GR=0;
function sizeCanvas(){
  const el=document.getElementById('ccol');
  CW=el.clientWidth||1;
  const bodyH=document.getElementById('body').clientHeight||1;
  CH=Math.max(60,bodyH-72);cv.width=CW;cv.height=CH;GR=Math.min(CH*0.46,CW*0.19);
}
function gauge(cx,cy,r,mn,mx,val,unit,col,redAt){
  const SA=.75*Math.PI,EA=2.25*Math.PI,ARC=1.5*Math.PI;
  const pct=Math.max(0,Math.min(1,(val-mn)/(mx-mn)));
  const va=SA+pct*ARC,tr=r*.80,lw=Math.max(3,r*.13);
  cx2.beginPath();cx2.arc(cx,cy,tr,SA,EA,false);cx2.strokeStyle='#15152a';cx2.lineWidth=lw;cx2.lineCap='round';cx2.stroke();
  if(redAt!==undefined){const rp=(redAt-mn)/(mx-mn);cx2.beginPath();cx2.arc(cx,cy,tr,SA+rp*ARC,EA,false);cx2.strokeStyle='#280808';cx2.lineWidth=lw;cx2.stroke();}
  cx2.beginPath();cx2.arc(cx,cy,tr,SA,va,false);cx2.strokeStyle=col;cx2.lineWidth=lw*.75;cx2.lineCap='round';cx2.stroke();
  const ex=cx+tr*Math.cos(va),ey=cy+tr*Math.sin(va);
  cx2.beginPath();cx2.arc(ex,ey,lw*.5,0,Math.PI*2);cx2.fillStyle=col;cx2.fill();
  cx2.textAlign='center';
  cx2.fillStyle='#e8e8e8';cx2.font=`bold ${r*.33}px Arial`;cx2.fillText(Math.round(val),cx,cy+r*.1);
  cx2.fillStyle='#3a3a3a';cx2.font=`${r*.17}px Arial`;cx2.fillText(unit,cx,cy+r*.32);
}
function drawDash(){
  const spd=tLive?tSpeed:simSpd,rpm=tLive?tRpm:simRpm,gear=tLive?tGear:localGear;
  cx2.clearRect(0,0,CW,CH);cx2.fillStyle='#07070f';cx2.fillRect(0,0,CW,CH);
  const cy=CH*.60,pad=GR*.15,lcx=GR+pad,rcx=CW-GR-pad,gcx=CW/2;
  gauge(lcx,cy,GR,0,160,spd,'km/h','#00c8ff',130);
  gauge(rcx,cy,GR,0,2500,rpm,'RPM','#ff4060',RED_RPM);
  cx2.textAlign='center';cx2.fillStyle=gear===0?'#444':'#fff';cx2.font=`bold ${GR*.75}px Arial`;
  cx2.fillText(gear===0?'N':String(gear),gcx,cy+GR*.05);
  cx2.fillStyle='#2a2a2a';cx2.font=`${GR*.17}px Arial`;cx2.fillText('GEAR',gcx,cy+GR*.3);
  cx2.strokeStyle='#181828';cx2.lineWidth=1;
  [lcx+GR*.92,rcx-GR*.92].forEach(x=>{cx2.beginPath();cx2.moveTo(x,CH*.1);cx2.lineTo(x,CH*.9);cx2.stroke();});
}
const tfill=document.getElementById('tfill'),tvalEl=document.getElementById('tval');
function updateTilt(){
  const abs=Math.abs(tilt);tvalEl.textContent=(tilt>=0?'+':'')+tilt.toFixed(1)+'°';
  if(tilt<-1)      tfill.style.cssText=`left:${50-abs/90*50}%;width:${abs/90*50}%;background:#4488ff`;
  else if(tilt>1)  tfill.style.cssText=`left:50%;width:${abs/90*50}%;background:#00e676`;
  else             tfill.style.cssText='width:0';
}
function sb2(t,s){if(t==='g'){gas=s;document.getElementById('gsb').classList.toggle('on',s)}if(t==='b'){brake=s;document.getElementById('bkb').classList.toggle('on',s)}}
function sh(s){horn=s;document.getElementById('hornb2').classList.toggle('on',s)}
function tapShift(d){
  if(d==='u'){shiftUp=true;flash('supb')}
  if(d==='d'){shiftDown=true;flash('sdnb')}
  if(ws&&ws.readyState===1){
    ws.send(JSON.stringify({tilt,gas,brake,horn,profile,shiftUp,shiftDown,indLeft,indRight,lights}));
    shiftUp=false;shiftDown=false;
  }
}
function tapInd(side){if(side==='l'){indLeft=true;flash('ind-l')}else{indRight=true;flash('ind-r')}}
function tapLights(){lights=true;flash('lightsb2')}
function flash(id){const el=document.getElementById(id);el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),220);}
document.querySelectorAll('.pb').forEach(b=>b.addEventListener('click',()=>{profile=+b.dataset.p;document.querySelectorAll('.pb').forEach(x=>x.classList.remove('on'));b.classList.add('on');}));
const dot=document.getElementById('dot'),lbl=document.getElementById('lbl'),tliveEl=document.getElementById('tlive');
let ws,sendLoop;
function connect(){
  ws=new WebSocket('wss://'+window.location.hostname+':'+WS_PORT);
  ws.onopen=()=>{dot.className='ok';lbl.textContent='Connected ✓';sendLoop=setInterval(sendCtrl,33);};
  ws.onclose=ws.onerror=()=>{dot.className='err';lbl.textContent='Reconnecting…';clearInterval(sendLoop);setTimeout(connect,2000);};
  ws.onmessage=e=>{
    try{const d=JSON.parse(e.data);if(d.t){tSpeed=d.t.speed||0;tRpm=d.t.rpm||800;tGear=d.t.gear||0;tLive=!!d.t.live;tliveEl.textContent=tLive?'● LIVE':'● SIM';tliveEl.className=tLive?'on':'';}}catch(err){}
  };
}
function sendCtrl(){
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({tilt,gas,brake,horn,profile,shiftUp,shiftDown,indLeft,indRight,lights}));
  indLeft=indRight=lights=false;
  shiftUp=shiftDown=false;
}
function getLandscapeAngle(){
  if(screen.orientation&&screen.orientation.angle!=null)return screen.orientation.angle;
  const wo=window.orientation||0;if(wo===-90)return 90;if(wo===90)return 270;return 0;
}
function onOri(e){
  const angle=getLandscapeAngle();
  let raw;
  if(angle===90)       raw=-(e.beta||0);   // top→right: negate beta for correct direction
  else if(angle===270) raw= (e.beta||0);   // top→left
  else                 raw= (e.gamma||0);  // portrait fallback
  tilt=Math.max(-90,Math.min(90,raw));
}
function frame(){trackLocalGear();if(!tLive)simTick();updateTilt();drawDash();requestAnimationFrame(frame);}
function onResize(){sizeCanvas();drawDash();}
window.addEventListener('resize',onResize);
window.addEventListener('orientationchange',()=>setTimeout(onResize,150));
const tapEl=document.getElementById('tap');
async function start(){
  if(typeof DeviceOrientationEvent!=='undefined'&&typeof DeviceOrientationEvent.requestPermission==='function'){
    try{const r=await DeviceOrientationEvent.requestPermission();if(r!=='granted'){alert('Sensor access denied. Enable Motion in Safari Settings.');return;}}catch(e){}
  }
  window.addEventListener('deviceorientation',onOri,{passive:true});
  tapEl.classList.add('hide');connect();sizeCanvas();frame();
}
tapEl.addEventListener('click',start);
tapEl.addEventListener('touchend',e=>{e.preventDefault();start();});
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

    def start(self, loop: asyncio.AbstractEventLoop):
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
            await self._telem_push()

    async def _ws_handler(self, ws):
        addr = ws.remote_address[0]
        print(f'[PHONE] Phone connected: {addr}')
        with self._lock: self._clients.add(ws)
        if self._on_connect: self._on_connect(addr)
        try:
            async for msg in ws:
                try:
                    d = json.loads(msg)
                    self._ctrl.update(
                        tilt       = float(d.get('tilt',      0)),
                        gas        = bool (d.get('gas',       False)),
                        brake      = bool (d.get('brake',     False)),
                        horn       = bool (d.get('horn',      False)),
                        profile    = int  (d.get('profile',   1)),
                        shift_up   = bool (d.get('shiftUp',   False)),
                        shift_down = bool (d.get('shiftDown', False)),
                        ind_left   = bool (d.get('indLeft',   False)),
                        ind_right  = bool (d.get('indRight',  False)),
                        lights     = bool (d.get('lights',    False)),
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

    async def _telem_push(self):
        while True:
            await asyncio.sleep(0.10)
            snap = _telemetry.snapshot
            msg  = json.dumps({'t': snap})
            with self._lock:
                dead = set()
                for ws in list(self._clients):
                    try: await ws.send(msg)
                    except Exception: dead.add(ws)
                self._clients.difference_update(dead)


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
        self._poll_telem()
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

        tk.Label(right, text='Funbit Telemetry Plugin',
                 bg=self.PANEL, fg=self.DIM, font=('Arial', 9, 'bold')).pack(anchor='w')

        self._telem_live_lbl = tk.Label(right, text='● Waiting for ETS2…',
                                         bg=self.PANEL, fg=self.DIM, font=('Arial', 9))
        self._telem_live_lbl.pack(anchor='w', pady=(2,10))

        info = [
            'One-time install (free):',
            'github.com/RenCloud/scs-sdk-plugin',
            '',
            'Copy Win64/scs-telemetry.dll to:',
            'Documents/ETS2/plugins/',
            '',
            'Launch ETS2 — gauges go live',
            'automatically. No extra server.',
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
        self._live_lbl.config(text='● LIVE' if live else '● SIM',
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
