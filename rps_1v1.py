"""
Rock Paper Scissors — 1v1 Split Camera Mode
============================================
One camera split in half:  LEFT = Player 1  |  RIGHT = Player 2

Game modes:
  - Best of 3  (first to win 2 rounds wins the match)
  - Unlimited  (play forever, manual reset)

Controls:
  SPACE  - start round
  M      - toggle mode (Best of 3 / Unlimited)
  R      - reset scores / new match
  Q      - quit

Run:
  python rps_1v1.py
"""

import cv2
import numpy as np
import time
import random
from pathlib import Path
import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import base64
# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed — running in DEMO mode.")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("rps_trained.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("rps_yolo.pt")

# ── Classes — match YOUR model's order exactly ────────────────────────────────
CLASSES = ["Paper", "Rock", "Scissors"]
WINS    = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_P1        = (255, 140,  50)   # orange  — Player 1
C_P2        = (80,  180, 255)   # blue    — Player 2
C_ACCENT    = (0,   210, 255)   # cyan
C_WIN       = (50,  220, 100)   # green
C_DRAW      = (180, 180,  50)   # yellow
C_DARK      = (25,   20,  40)
C_PANEL     = (18,   14,  30)
C_WHITE     = (255, 255, 255)
C_DIVIDER   = (60,   55,  90)

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_SIMPLEX

# ── Game modes ────────────────────────────────────────────────────────────────
MODE_BO3       = "Best of 3"
MODE_UNLIMITED = "Unlimited"


# ── Helpers ───────────────────────────────────────────────────────────────────
def rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    x1, y1 = pt1; x2, y2 = pt2
    ov = img.copy()
    cv2.rectangle(ov, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(ov, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                   (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(ov, (cx,cy), radius, color, thickness)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def text_center(img, text, cx, cy, font, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx-tw//2, cy+th//2), font, scale, color, thickness, cv2.LINE_AA)


def draw_pip_bars(img, wins_p1, wins_p2, target, x, y, w):
    """Draw round-win pips (● circles) for Best of 3."""
    pip_r = 10
    gap   = 28
    total = target * 2 - 1   # max rounds (e.g. 3 for BO3)
    mid   = x + w // 2

    # Left pips  (P1 wins) — drawn right-to-left from center
    for i in range(target - 1, -1, -1):
        cx = mid - (target - i) * gap
        filled = i < wins_p1
        cv2.circle(img, (cx, y), pip_r, C_P1, -1 if filled else 2)

    # Right pips (P2 wins) — drawn left-to-right from center
    for i in range(target):
        cx = mid + (i + 1) * gap
        filled = i < wins_p2
        cv2.circle(img, (cx, y), pip_r, C_P2, -1 if filled else 2)


# ── YOLO detector ─────────────────────────────────────────────────────────────
class Detector:
    def __init__(self):
        self.model     = None
        self.demo_mode = not YOLO_AVAILABLE
        if YOLO_AVAILABLE and MODEL_PATH.exists():
            try:
                self.model = YOLO(str(MODEL_PATH))
                print(f"✅ Model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[ERROR] {e}")
                self.demo_mode = True
        else:
            self.demo_mode = True
        if self.demo_mode:
            print("⚠️  DEMO mode — moves are random.")

    def detect(self, roi):
        """Detect move in a cropped ROI image."""
        if self.demo_mode:
            return random.choice(CLASSES), 0.85
        results = self.model(roi, verbose=False, imgsz=320)[0]
        if not results.boxes or len(results.boxes) == 0:
            return None, 0.0
        best = max(results.boxes, key=lambda b: float(b.conf))
        cls  = int(best.cls)
        conf = float(best.conf)
        return (CLASSES[cls], conf) if cls < len(CLASSES) else (None, 0.0)


# ── 1v1 Game ──────────────────────────────────────────────────────────────────
class RPS1v1:
    COUNTDOWN_SECS  = 3
    RESULT_DURATION = 2.0
    WIN_DURATION    = 4.0    # match winner screen

    def __init__(self, detector: Detector):
        self.det   = detector
        self.mode  = MODE_BO3
        self._reset_match()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_match(self):
        self.score       = {"p1": 0, "p2": 0, "draw": 0}
        self.round_wins  = {"p1": 0, "p2": 0}   # within current match (BO3)
        self.round_num   = 1
        self.state       = "WAITING"             # WAITING|COUNTDOWN|RESULT|MATCH_WIN
        self.p1_move     = None
        self.p2_move     = None
        self.round_result = ""
        self.round_color  = C_WHITE
        self.match_winner = None
        self.timer_start  = 0

    def toggle_mode(self):
        self.mode = MODE_UNLIMITED if self.mode == MODE_BO3 else MODE_BO3
        self._reset_match()

    def start_round(self):
        if self.state != "WAITING":
            return
        self.state       = "COUNTDOWN"
        self.timer_start = time.time()
        self.p1_move = self.p2_move = None

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def _evaluate(self, m1, m2):
        if m1 == m2:
            return "DRAW", C_DRAW, None
        winner = "p1" if WINS[m1] == m2 else "p2"
        label  = "PLAYER 1 WINS ROUND! 🎉" if winner == "p1" else "PLAYER 2 WINS ROUND! 🎉"
        color  = C_P1 if winner == "p1" else C_P2
        return label, color, winner

    # ── Tick ──────────────────────────────────────────────────────────────────
    def update(self, frame):
        h, w = frame.shape[:2]
        now  = time.time()

        # Split camera — left half = P1, right half = P2
        mid      = w // 2
        roi_p2   = frame[:, :mid]
        roi_p1   = frame[:, mid:]

        live_p1, conf_p1 = self.det.detect(roi_p1)
        live_p2, conf_p2 = self.det.detect(roi_p2)

        # ── State machine ─────────────────────────────────────────────────────
        if self.state == "COUNTDOWN":
            elapsed = now - self.timer_start
            if elapsed >= self.COUNTDOWN_SECS:
                # Lock moves
                self.p1_move = live_p1 or random.choice(CLASSES)
                self.p2_move = live_p2 or random.choice(CLASSES)
                label, color, winner = self._evaluate(self.p1_move, self.p2_move)
                self.round_result = label
                self.round_color  = color

                if winner is None:
                    self.score["draw"] += 1
                else:
                    self.score[winner] += 1
                    self.round_wins[winner] += 1

                # Check match winner (BO3 = first to 2)
                target = 2 if self.mode == MODE_BO3 else None
                if target and (self.round_wins["p1"] >= target or
                               self.round_wins["p2"] >= target):
                    self.match_winner = "p1" if self.round_wins["p1"] >= target else "p2"
                    self.state        = "MATCH_WIN"
                else:
                    self.round_num += 1
                    self.state      = "RESULT"

                self.timer_start = now

        elif self.state == "RESULT":
            if now - self.timer_start > self.RESULT_DURATION:
                self.state = "WAITING"

        elif self.state == "MATCH_WIN":
            if now - self.timer_start > self.WIN_DURATION:
                # Auto-reset round wins but keep total scores
                self.round_wins = {"p1": 0, "p2": 0}
                self.round_num  = 1
                self.match_winner = None
                self.state = "WAITING"

        # ── Draw ──────────────────────────────────────────────────────────────
        canvas = self._draw(frame, live_p1, conf_p1, live_p2, conf_p2, now)
        return canvas

    # ── Draw UI ───────────────────────────────────────────────────────────────
    def _draw(self, frame, lm1, c1, lm2, c2, now):
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        mid   = w // 2

        # After flip: left = P2 feed, right = P1 feed — swap ROIs visually
        # (flip mirrors camera so left side of flipped = right of original = P2)
        # We keep display natural: label left panel P1, right panel P2
        # but detect from original pre-flip ROIs (done in update before this call)

        # Subtle dark overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(w,h),(0,0,0),-1)
        frame = cv2.addWeighted(ov, 0.18, frame, 0.82, 0)

        # ── Center divider ────────────────────────────────────────────────────
        cv2.rectangle(frame, (mid-2, 0), (mid+2, h), C_DIVIDER, -1)

        # ── Top bar ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0,0),(w,68), C_PANEL, -1)
        text_center(frame, "ROCK · PAPER · SCISSORS  —  1 v 1",
                    w//2, 28, FONT_BOLD, 0.8, C_ACCENT, 2)

        # Mode badge
        rounded_rect(frame, (w-170, 8), (w-8, 56), C_DARK, radius=8, alpha=0.9)
        text_center(frame, self.mode, w-89, 32, FONT, 0.55, C_ACCENT, 1)

        # Round indicator
        if self.mode == MODE_BO3:
            text_center(frame, f"Round {self.round_num} of 3",
                        w//2, 52, FONT, 0.5, (160,160,160), 1)

        # ── Score cards ───────────────────────────────────────────────────────
        #  P1 score (left)
        rounded_rect(frame, (10, 78), (210, 148), C_DARK, radius=10, alpha=0.85)
        text_center(frame, "PLAYER 1", 110, 95,  FONT, 0.55, C_P1, 1)
        text_center(frame, str(self.score["p1"]), 110, 128, FONT_BOLD, 1.4, C_WHITE, 2)

        # Draw (center)
        rounded_rect(frame, (w//2-70, 78), (w//2+70, 148), C_DARK, radius=10, alpha=0.85)
        text_center(frame, "DRAW", w//2, 95,  FONT, 0.55, C_DRAW, 1)
        text_center(frame, str(self.score["draw"]), w//2, 128, FONT_BOLD, 1.4, C_WHITE, 2)

        # P2 score (right)
        rounded_rect(frame, (w-210, 78), (w-10, 148), C_DARK, radius=10, alpha=0.85)
        text_center(frame, "PLAYER 2", w-110, 95,  FONT, 0.55, C_P2, 1)
        text_center(frame, str(self.score["p2"]), w-110, 128, FONT_BOLD, 1.4, C_WHITE, 2)

        # ── BO3 pip bars ──────────────────────────────────────────────────────
        if self.mode == MODE_BO3:
            draw_pip_bars(frame, self.round_wins["p1"], self.round_wins["p2"],
                          2, 0, 163, w)

        # ── Player panels ─────────────────────────────────────────────────────
        panel_y = 178
        panel_h = h - panel_y - 100
        # P1 panel (left half)
        rounded_rect(frame, (10, panel_y), (mid-15, panel_y+panel_h),
                     C_DARK, radius=16, alpha=0.55)
        text_center(frame, "◀  PLAYER 1", mid//2, panel_y+28, FONT, 0.7, C_P1, 1)

        # P2 panel (right half)
        rounded_rect(frame, (mid+15, panel_y), (w-10, panel_y+panel_h),
                     C_DARK, radius=16, alpha=0.55)
        text_center(frame, "PLAYER 2  ▶", mid + (w-mid)//2, panel_y+28, FONT, 0.7, C_P2, 1)

        my = panel_y + panel_h//2 + 20

        # ── State content ─────────────────────────────────────────────────────
        if self.state == "WAITING":
            self._draw_live(frame, lm1, c1, mid//2,          my, C_P1)
            self._draw_live(frame, lm2, c2, mid+(w-mid)//2,  my, C_P2)

            # Instruction banner
            rounded_rect(frame, (w//2-220, h-95), (w//2+220, h-55),
                         C_ACCENT, radius=12, alpha=0.2)
            cv2.rectangle(frame, (w//2-220, h-95),(w//2+220, h-55), C_ACCENT, 1)
            text_center(frame, "Press SPACE to start round",
                        w//2, h-75, FONT, 0.75, C_WHITE, 1)

        elif self.state == "COUNTDOWN":
            elapsed   = now - self.timer_start
            remaining = max(0, self.COUNTDOWN_SECS - elapsed)
            cnt       = str(int(remaining)+1) if remaining > 0.05 else "GO!"
            progress  = elapsed / self.COUNTDOWN_SECS

            # Show live moves during countdown
            self._draw_live(frame, lm1, c1, mid//2,         my, C_P1)
            self._draw_live(frame, lm2, c2, mid+(w-mid)//2, my, C_P2)

            # Countdown circle in center
            cx, cy, r = w//2, my, 52
            cv2.circle(frame, (cx,cy), r+4, C_DARK, -1)
            end_ang = int(-90 + progress*360)
            cv2.ellipse(frame, (cx,cy),(r,r), 0, -90, end_ang, C_ACCENT, 6)
            text_center(frame, cnt, cx, cy, FONT_BOLD, 1.6, C_WHITE, 3)

        elif self.state == "RESULT":
            m1 = self.p1_move or "?"
            m2 = self.p2_move or "?"
            text_center(frame, m1.upper(), mid//2,         my, FONT_BOLD, 1.6, C_P1,  2)
            text_center(frame, m2.upper(), mid+(w-mid)//2, my, FONT_BOLD, 1.6, C_P2,  2)
            text_center(frame, "VS", w//2, my, FONT_BOLD, 1.0, C_ACCENT, 2)

            # Result banner
            bw, bh = 480, 56
            bx, by = w//2-bw//2, panel_y+panel_h+10
            rounded_rect(frame, (bx,by),(bx+bw,by+bh),
                         self.round_color, radius=14, alpha=0.3)
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh), self.round_color, 2)
            text_center(frame, self.round_result, w//2, by+bh//2,
                        FONT_BOLD, 0.9, self.round_color, 2)

        elif self.state == "MATCH_WIN":
            mw = self.match_winner
            col  = C_P1 if mw == "p1" else C_P2
            name = "PLAYER 1" if mw == "p1" else "PLAYER 2"

            # Big winner overlay
            ov2 = frame.copy()
            cv2.rectangle(ov2,(0,0),(w,h),(0,0,0),-1)
            frame = cv2.addWeighted(ov2, 0.45, frame, 0.55, 0)

            text_center(frame, "🏆  MATCH WINNER  🏆",
                        w//2, h//2-80, FONT_BOLD, 1.0, C_ACCENT, 2)
            text_center(frame, name,
                        w//2, h//2,   FONT_BOLD, 2.8, col,     3)
            text_center(frame, f"{self.round_wins[mw]} - {self.round_wins['p2' if mw=='p1' else 'p1']}",
                        w//2, h//2+80, FONT_BOLD, 1.4, C_WHITE, 2)
            text_center(frame, "New match starting soon…",
                        w//2, h//2+130, FONT, 0.65, (160,160,160), 1)

        # ── ROI guide boxes ───────────────────────────────────────────────────
        if self.state in ("WAITING", "COUNTDOWN"):
            roi_col_p1 = C_P1 if lm1 else (80, 60, 100)
            roi_col_p2 = C_P2 if lm2 else (80, 60, 100)
            # P1 box — left side (after flip, right of original)
            bsize = min(panel_h-60, mid-30)
            bx1 = mid//2 - bsize//2
            by1 = panel_y + 55
            cv2.rectangle(frame, (bx1, by1), (bx1+bsize, by1+bsize), roi_col_p1, 2)
            # P2 box — right side
            bx2 = mid + (w-mid)//2 - bsize//2
            cv2.rectangle(frame, (bx2, by1), (bx2+bsize, by1+bsize), roi_col_p2, 2)

        # ── Demo badge ────────────────────────────────────────────────────────
        if self.det.demo_mode:
            text_center(frame, "DEMO", 50, 58, FONT, 0.45, (0,140,255), 1)

        # ── Bottom hint ───────────────────────────────────────────────────────
        hint = "SPACE start  |  M toggle mode  |  R reset  |  Q quit"
        text_center(frame, hint, w//2, h-18, FONT, 0.45, (110,110,110), 1)

        return frame

    def _draw_live(self, img, move, conf, cx, cy, color):
        disp = (move or "?").upper()
        text_center(img, disp, cx, cy, FONT_BOLD, 1.6, color, 2)
        if move and conf:
            text_center(img, f"{conf:.0%}", cx, cy+45, FONT, 0.55, C_WHITE, 1)

def set_background(image_file):
  
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
set_background(r"image\bg.jpeg")

# st.set_page_config(page_title="Rock-paper-scissors Real-Time Game", layout="centered")

# st.title("🖐 Rock Paper Scissors - Real Time Game")
# st.markdown("Press **Play** to start camera and run YOLO detection")


# play = st.button("▶ Play")
# stop = st.button("⏹ Stop")

frame_placeholder = st.empty()
# ── Entry point ───────────────────────────────────────────────────────────────
# def main():
#     if play:
#         cap = cv2.VideoCapture(1)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

#         det  = Detector()
#         game = RPS1v1(det)
        
#         print("\n🎮 1v1 Rock Paper Scissors")
#         print("   Left half  = Player 1")
#         print("   Right half = Player 2")
#         print("\n   SPACE  - start round")
#         print("   M      - toggle Best of 3 / Unlimited")
#         print("   R      - reset")
#         print("   Q      - quit\n")
        
#         if not cap.isOpened():
#             st.error("❌ Cannot open webcam")
#             return

        

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             canvas = game.update(frame)
#             # cv2.imshow("RPS 1v1 — YOLO Edition", canvas)
#             frame_placeholder.image(
#             canvas,
#             channels="RGB",
#             # use_container_width=True
#         )
            
#             key = cv2.waitKey(1) & 0xFF
#             if   key == ord('q'):
#                 break
#             elif key == ord(' '):
#                 game.start_round()
#             elif key == ord('m'):
#                 game.toggle_mode()
#                 print(f"Mode switched → {game.mode}")
#             elif key == ord('r'):
#                 game._reset_match()
#                 print("Match reset.")
           
                

#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
if "game" not in st.session_state:
    st.session_state.game = RPS1v1(Detector())

if "running" not in st.session_state:
    st.session_state.running = False
def main():
    st.set_page_config(
        page_title="Rock Paper Scissors - 1v1",
        layout="wide"
    )

    st.title("🖐 Rock Paper Scissors — 1v1 YOLO")

    # ---------- Session State ----------
    if "game" not in st.session_state:
        st.session_state.game = RPS1v1(Detector())

    if "cap" not in st.session_state:
        st.session_state.cap = None

    if "running" not in st.session_state:
        st.session_state.running = False

    # ---------- Controls ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎥 Start Camera"):
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.running = True

    with col2:
        if st.button("▶ Start Round"):
            st.session_state.game.start_round()

    with col3:
        if st.button("🔁 Toggle Mode"):
            st.session_state.game.toggle_mode()

    with col4:
        if st.button("🔄 Reset"):
            st.session_state.game._reset_match()

    # ---------- Frame Display ----------
    frame_placeholder = st.empty()

    if st.session_state.running and st.session_state.cap:
        ret, frame = st.session_state.cap.read()

        if not ret:
            st.error("❌ Cannot read from camera")
            return
        while st.session_state.running:
            ret, frame = st.session_state.cap.read()
            if not ret:
                break

            canvas = st.session_state.game.update(frame)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                canvas,
                channels="RGB",
                use_container_width=800,
                width=1000
            )

            time.sleep(0.01)  # FPS control ~60
main()