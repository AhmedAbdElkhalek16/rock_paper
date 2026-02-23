"""
Rock Paper Scissors Game using YOLOv8 hand gesture detection.
Run: python rps_game.py

Requirements:
    pip install ultralytics opencv-python numpy

The script downloads a pretrained RPS YOLO model automatically on first run.
"""

import cv2
import numpy as np
import time
import random
from pathlib import Path

# ── Try importing ultralytics ────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Running in DEMO mode (random detection).")
    print("  Install with: pip install ultralytics")


# ── Constants ────────────────────────────────────────────────────────────────
CLASSES = ["paper", "rock", "scissors"]
WINS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
EMOJI   = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}

# Model priority: 1) locally trained  2) downloaded pretrained  3) demo mode
MODEL_TRAINED   = Path("rps_trained.pt")          # output of 3_train.py
MODEL_PRETRAINED = Path("rps_yolo.pt")             # fallback download
MODEL_URL = "https://github.com/niconielsen32/ComputerVision/raw/master/RockPaperScissors/best.pt"

def resolve_model_path():
    if MODEL_TRAINED.exists():
        print(f"✅ Using locally trained model: {MODEL_TRAINED}")
        return MODEL_TRAINED
    print(f"ℹ️  Local model not found ({MODEL_TRAINED}). Falling back to pretrained.")
    return MODEL_PRETRAINED

MODEL_PATH = resolve_model_path()

# UI Colours  (BGR)
C_BG        = (15, 10, 25)
C_ACCENT    = (0, 210, 255)       # electric cyan
C_WIN       = (50, 220, 100)
C_LOSE      = (50,  80, 220)
C_DRAW      = (180, 180,  50)
C_WHITE     = (255, 255, 255)
C_DARK      = ( 30,  30,  50)
C_PANEL     = ( 25,  20,  40)

FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_SIMPLEX


# ── Helper: download model ────────────────────────────────────────────────────
def download_model():
    if MODEL_PATH.exists():
        return True
    print(f"Downloading RPS model weights to {MODEL_PATH} …")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Could not download model: {e}")
        print("  Falling back to DEMO mode.")
        return False


# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=1.0):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def put_text_centered(img, text, cx, y, font, scale, color, thickness=1):
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - w//2, y + h//2), font, scale, color, thickness, cv2.LINE_AA)


def draw_move_icon(img, move, cx, cy, size=60):
    """Draw ASCII art move icon."""
    icons = {
        "rock":     ["  ___  ", " /   \\ ", "|     |", " \\___/ "],
        "paper":    ["[=====]", "|     |", "|     |", "[=====]"],
        "scissors": [" /\\  /\\", "/  \\/  \\", "\\      /", " \\____/"],
        "?":        ["  ___  ", " / ? \\ ", "|  ?  |", " \\___/ "],
    }
    lines = icons.get(move, icons["?"])
    lh = 14
    for i, line in enumerate(lines):
        y = cy - len(lines)*lh//2 + i*lh
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, 0.9, 1)
        cv2.putText(img, line, (cx - w//2, y), cv2.FONT_HERSHEY_PLAIN,
                    0.9, C_WHITE, 1, cv2.LINE_AA)


# ── Main Game Class ──────────────────────────────────────────────────────────
class RPSGame:
    def __init__(self):
        self.score  = {"player": 0, "computer": 0, "draw": 0}
        self.state  = "WAITING"   # WAITING | COUNTDOWN | RESULT
        self.countdown_start = 0
        self.countdown_secs  = 3
        self.player_move  = None
        self.computer_move = None
        self.result_text  = ""
        self.result_color = C_WHITE
        self.result_start = 0
        self.result_duration = 2.5   # seconds to show result

        # YOLO
        self.model = None
        self.demo_mode = not YOLO_AVAILABLE
        if YOLO_AVAILABLE:
            ok = download_model()
            if ok:
                try:
                    self.model = YOLO(str(MODEL_PATH))
                    print("YOLO model loaded.")
                except Exception as e:
                    print(f"[ERROR] Loading model: {e}")
                    self.demo_mode = True
            else:
                self.demo_mode = True
        if self.demo_mode:
            print("Running in DEMO mode — moves are randomly simulated.")

    def detect_move(self, frame):
        if self.demo_mode:
            return random.choice(CLASSES), 0.9
        results = self.model(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None, 0.0
        best = max(results.boxes, key=lambda b: float(b.conf))
        cls  = int(best.cls)
        conf = float(best.conf)
        if cls < len(CLASSES):
            return CLASSES[cls], conf
        return None, 0.0

    def start_round(self):
        self.state = "COUNTDOWN"
        self.countdown_start = time.time()
        self.player_move  = None
        self.computer_move = None

    def evaluate(self, player, computer):
        if player == computer:
            return "DRAW", C_DRAW
        if WINS[player] == computer:
            return "YOU WIN! 🎉", C_WIN
        return "COMPUTER WINS", C_LOSE

    def update(self, frame):
        """Run one game tick. Returns annotated frame."""
        h, w = frame.shape[:2]
        now  = time.time()

        # ── Detect move every frame for live preview ─────────────────────────
        live_move, live_conf = self.detect_move(frame)

        # ── State machine ─────────────────────────────────────────────────────
        if self.state == "COUNTDOWN":
            elapsed = now - self.countdown_start
            remaining = self.countdown_secs - elapsed
            if remaining <= 0:
                # Lock in player move
                self.player_move   = live_move if live_move else random.choice(CLASSES)
                self.computer_move = random.choice(CLASSES)
                text, color        = self.evaluate(self.player_move, self.computer_move)
                self.result_text   = text
                self.result_color  = color
                if text == "DRAW":
                    self.score["draw"] += 1
                elif "YOU" in text:
                    self.score["player"] += 1
                else:
                    self.score["computer"] += 1
                self.state        = "RESULT"
                self.result_start = now

        elif self.state == "RESULT":
            if now - self.result_start > self.result_duration:
                self.state = "WAITING"

        # ── Draw UI ───────────────────────────────────────────────────────────
        canvas = self.draw_ui(frame, live_move, live_conf, now)
        return canvas

    def draw_ui(self, frame, live_move, live_conf, now):
        h, w = frame.shape[:2]

        # Mirror + darken
        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # ── Top bar ───────────────────────────────────────────────────────────
        draw_rounded_rect(frame, (0,0), (w, 70), C_PANEL, radius=0, alpha=0.9)
        title = "ROCK · PAPER · SCISSORS"
        put_text_centered(frame, title, w//2, 35, FONT_BOLD, 0.8, C_ACCENT, 2)

        # ── Score panel ───────────────────────────────────────────────────────
        sw = 220
        sx = (w - sw*3 - 40) // 2
        for i, (label, key, col) in enumerate([
            ("YOU",      "player",   C_WIN),
            ("DRAW",     "draw",     C_DRAW),
            ("COMPUTER", "computer", C_LOSE),
        ]):
            bx = sx + i*(sw+20)
            draw_rounded_rect(frame, (bx, 80), (bx+sw, 160), C_DARK, radius=12, alpha=0.85)
            put_text_centered(frame, label,                  bx+sw//2, 100,  FONT,      0.6, col, 1)
            put_text_centered(frame, str(self.score[key]),   bx+sw//2, 135,  FONT_BOLD, 1.4, C_WHITE, 2)

        # ── Move display area ─────────────────────────────────────────────────
        panel_y = 175
        panel_h = h - panel_y - 120
        # Player panel (left)
        draw_rounded_rect(frame, (20, panel_y), (w//2-10, panel_y+panel_h),
                          C_DARK, radius=18, alpha=0.6)
        put_text_centered(frame, "YOUR MOVE", w//4, panel_y+25, FONT, 0.65, C_ACCENT, 1)

        # Computer panel (right)
        draw_rounded_rect(frame, (w//2+10, panel_y), (w-20, panel_y+panel_h),
                          C_DARK, radius=18, alpha=0.6)
        put_text_centered(frame, "COMPUTER", 3*w//4, panel_y+25, FONT, 0.65, C_LOSE, 1)

        # ── State-specific content ────────────────────────────────────────────
        my = panel_y + panel_h//2 + 20

        if self.state == "WAITING":
            # Show live detected move
            disp = live_move if live_move else "?"
            put_text_centered(frame, disp.upper() if disp != "?" else "?",
                              w//4, my, FONT_BOLD, 1.5, C_WHITE, 2)
            if live_move:
                conf_txt = f"conf: {live_conf:.0%}"
                put_text_centered(frame, conf_txt, w//4, my+40, FONT, 0.5, C_ACCENT, 1)
            put_text_centered(frame, "...", 3*w//4, my, FONT_BOLD, 1.5, C_DARK, 2)

            # Instruction
            draw_rounded_rect(frame, (w//2-200, h-110), (w//2+200, h-70),
                              C_ACCENT, radius=12, alpha=0.25)
            put_text_centered(frame, "Press SPACE to start round", w//2, h-90,
                              FONT, 0.7, C_WHITE, 1)

        elif self.state == "COUNTDOWN":
            elapsed  = now - self.countdown_start
            remaining = max(0, self.countdown_secs - elapsed)
            cnt_text = str(int(remaining) + 1) if remaining > 0 else "GO!"
            # Show live move
            disp = live_move if live_move else "?"
            put_text_centered(frame, disp.upper() if disp != "?" else "?",
                              w//4, my, FONT_BOLD, 1.5, C_WHITE, 2)
            # Countdown circle
            cx, cy, r = 3*w//4, my, 55
            progress = elapsed / self.countdown_secs
            end_angle = int(-90 + progress * 360)
            cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, end_angle, C_ACCENT, 6)
            put_text_centered(frame, cnt_text, cx, cy, FONT_BOLD, 1.6, C_WHITE, 3)

        elif self.state == "RESULT":
            pm = self.player_move   or "?"
            cm = self.computer_move or "?"
            put_text_centered(frame, pm.upper(), w//4,    my, FONT_BOLD, 1.5, C_WHITE, 2)
            put_text_centered(frame, cm.upper(), 3*w//4,  my, FONT_BOLD, 1.5, C_WHITE, 2)

            # VS divider
            put_text_centered(frame, "VS", w//2, my, FONT_BOLD, 1.0, C_ACCENT, 2)

            # Result banner
            bw, bh = 420, 55
            bx, by = w//2 - bw//2, panel_y + panel_h + 15
            draw_rounded_rect(frame, (bx, by), (bx+bw, by+bh),
                              self.result_color, radius=14, alpha=0.3)
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), self.result_color, 2)
            put_text_centered(frame, self.result_text, w//2, by+bh//2,
                              FONT_BOLD, 0.95, self.result_color, 2)

        # ── Demo mode badge ───────────────────────────────────────────────────
        if self.demo_mode:
            badge = "DEMO MODE"
            (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.5, 1)
            cv2.rectangle(frame, (w-bw-20, 75), (w-5, 75+bh+8), (60,30,0), -1)
            cv2.putText(frame, badge, (w-bw-12, 75+bh+2), FONT, 0.5, (0,160,255), 1, cv2.LINE_AA)

        # ── Quit hint ─────────────────────────────────────────────────────────
        put_text_centered(frame, "Q quit  |  R reset  |  SPACE new round",
                          w//2, h-15, FONT, 0.45, (120,120,120), 1)

        return frame


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    game = RPSGame()
    print("\nControls:  SPACE = start round   R = reset score   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = game.update(frame)
        cv2.imshow("Rock · Paper · Scissors  (YOLO Edition)", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and game.state == "WAITING":
            game.start_round()
        elif key == ord('r'):
            game.score = {"player": 0, "computer": 0, "draw": 0}
            game.state = "WAITING"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
