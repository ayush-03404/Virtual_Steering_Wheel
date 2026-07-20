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
DEAD_ZONE_DEG       = 2       # tilt within this = no steering output  (was 14 — smaller so tiny tilts register)
STEERING_CURVE      = 1.5     # 1=linear  2=quadratic  3=cubic  (cubic = very gentle at small angles, progressive)
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
# soft_zone = tilt angle (°) that maps to full lock — lower = less sensitive
SPEED_PROFILES = {
    ord('1'): dict(name="CITY",      soft_zone=65),   # was 28 — wider so same tilt = gentler response
    ord('2'): dict(name="HIGHWAY",   soft_zone=45),   # was 26
    ord('3'): dict(name="MOTORWAY",  soft_zone=28),   # was 16
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
