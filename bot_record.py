import os
import time
import numpy as np
import cv2
from mss import mss
from ultralytics import YOLO
from pynput.keyboard import Controller, Key

# ---------------- Config ----------------
ROI = {"left": 68, "top": 175, "width": 484, "height": 194}
MODEL_PATH = "runs/detect/train2/weights/best.pt"
CONF = 0.35

# Timing (TTI)
TTI_JUMP = 0.33
TTI_DUCK = 0.26
MIN_SPEED = 60.0
MAX_TTI = 1.2

# Fallback pixel triggers
PIX_TRIGGER_CACTUS = 135
PIX_TRIGGER_BIRD = 165

# Anti-spam jump
JUMP_COOLDOWN = 0.22
AIRBORNE_LOCK = 0.28

# Emergency override (pour les birds LOW quand lock bloque)
EMERGENCY_TTI = 0.08   # s
EMERGENCY_DIST = 35    # px

# Duck deadline-based (non-blocking)
DUCK_PAD = 0.02
DUCK_HOLD_MIN = 0.10
DUCK_HOLD_MAX = 0.45
DUCK_COOLDOWN = 0.0

# X hit points
LEAD_FRAC_CACTUS = 0.72
LEAD_FRAC_BIRD = 0.20

# Dino head line (from bottom, stable)
HEAD_FROM_BOTTOM = 0.78

# Names
NAME_CACTUS = "cactus"
NAME_BIRD = "bird"
NAME_DINO = "dino"

# Colors (BGR)
COLOR_CACTUS = (0, 255, 0)     # green
COLOR_BIRD = (255, 0, 0)       # blue
COLOR_DINO = (0, 255, 255)     # yellow

# ---------------- Keyboard ----------------
kb = Controller()

def tap_space():
    kb.press(Key.space)
    kb.release(Key.space)

def press_down():
    kb.press(Key.down)

def release_down():
    kb.release(Key.down)

# ---------------- Helpers ----------------
def iter_boxes(res):
    return [] if res.boxes is None else list(res.boxes)

def pick_best_box_by_conf(boxes, target_cls):
    best = None
    for b in boxes:
        cls = int(b.cls[0].item())
        if cls != target_cls:
            continue
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        cand = (x1, y1, x2, y2, conf)
        if best is None or conf > best[4]:
            best = cand
    return best

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def draw_danger_zone(vis, x_start, x_end, alpha=0.16):
    h, w = vis.shape[:2]
    x_start = max(0, min(w - 1, int(x_start)))
    x_end = max(0, min(w, int(x_end)))
    if x_end <= x_start:
        return
    overlay = vis.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, h - 1), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

def open_video_writer(out_path: str, w: int, h: int, fps: int):
    # Try mp4v -> avc1 -> XVID(.avi fallback)
    fourccs = [("mp4v", out_path), ("avc1", out_path)]
    for fourcc, path in fourccs:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened():
            return vw, path

    # Fallback AVI
    avi_path = os.path.splitext(out_path)[0] + ".avi"
    vw = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if vw.isOpened():
        return vw, avi_path

    raise RuntimeError("Impossible d'ouvrir VideoWriter (codec). Essaie d'installer une build OpenCV avec support mp4/avc1.")

# ---------------- Main ----------------
def main():
    model = YOLO(MODEL_PATH)
    names = model.names

    # class indices
    dino_cls = cactus_cls = bird_cls = None
    for k, v in names.items():
        if v == NAME_DINO:
            dino_cls = k
        elif v == NAME_CACTUS:
            cactus_cls = k
        elif v == NAME_BIRD:
            bird_cls = k
    if None in (dino_cls, cactus_cls, bird_cls):
        raise RuntimeError("Classes cactus/bird/dino introuvables dans model.names.")

    last_jump_t = 0.0
    jump_lock_until = 0.0
    last_duck_t = 0.0

    # speed EMA (px/s)
    prev_cx = None
    prev_t = None
    speed_px_s = None

    # duck deadline
    down_held = False
    duck_until = 0.0

    # Video output
    os.makedirs("recordings", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("recordings", f"dino_{ts}.mp4")
    fps_out = 30
    vw, real_out_path = open_video_writer(out_path, ROI["width"], ROI["height"], fps_out)

    with mss() as sct:
        cv2.namedWindow("Dino Bot (REC)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dino Bot (REC)", 980, 380)

        try:
            while True:
                now = time.time()

                # release duck if time over
                if down_held and now >= duck_until:
                    release_down()
                    down_held = False
                    duck_until = 0.0

                img = np.array(sct.grab(ROI))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                res = model.predict(frame, conf=CONF, verbose=False)[0]
                boxes = iter_boxes(res)

                # dino
                dino_box = pick_best_box_by_conf(boxes, dino_cls)

                vis = frame.copy()

                if dino_box is None:
                    # still record raw frame with no text
                    vw.write(vis)
                    cv2.imshow("Dino Bot (REC)", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                    continue

                dx1, dy1, dx2, dy2, _ = dino_box
                dino_front_x = dx2
                dino_h = max(5.0, dy2 - dy1)
                head_y = dy2 - HEAD_FROM_BOTTOM * dino_h

                # draw dino box (yellow)
                cv2.rectangle(vis, (int(dx1), int(dy1)), (int(dx2), int(dy2)), COLOR_DINO, 2)

                # obstacles list (ahead test uses x2)
                obs = []
                for b in boxes:
                    cls = int(b.cls[0].item())
                    if cls not in (cactus_cls, bird_cls):
                        continue

                    name = names.get(cls, str(cls))
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

                    if x2 <= dino_front_x:
                        continue

                    lead = LEAD_FRAC_CACTUS if name == NAME_CACTUS else LEAD_FRAC_BIRD
                    x_hit = x1 + lead * (x2 - x1)
                    dist_hit = x_hit - dino_front_x
                    if dist_hit > ROI["width"] * 0.95:
                        continue

                    cx = (x1 + x2) / 2.0
                    obs.append((name, x1, y1, x2, y2, x_hit, dist_hit, cx))

                    # draw all boxes with requested colors (no text)
                    color = COLOR_CACTUS if name == NAME_CACTUS else COLOR_BIRD
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                if not obs:
                    vw.write(vis)
                    cv2.imshow("Dino Bot (REC)", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                    continue

                # choose target: smallest positive dist_hit
                obs.sort(key=lambda o: (o[6] if o[6] >= 0 else 1e9))
                name, x1, y1, x2, y2, x_hit, dist_hit, cx = obs[0]

                # update speed EMA
                if prev_cx is not None and prev_t is not None:
                    dt = now - prev_t
                    if dt > 1e-3:
                        v = (prev_cx - cx) / dt
                        if v > MIN_SPEED:
                            speed_px_s = v if speed_px_s is None else 0.75 * speed_px_s + 0.25 * v
                prev_cx = cx
                prev_t = now

                tti = None
                if speed_px_s is not None and dist_hit > 0:
                    tti = dist_hit / speed_px_s

                # danger zone (no text)
                if speed_px_s is not None:
                    dz = min(ROI["width"] - 1, int(speed_px_s * TTI_JUMP))
                    draw_danger_zone(vis, dino_front_x, dino_front_x + dz)

                # decisions
                if name == NAME_CACTUS:
                    urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_CACTUS)
                    if urgent and now >= jump_lock_until and (now - last_jump_t) >= JUMP_COOLDOWN:
                        tap_space()
                        last_jump_t = now
                        jump_lock_until = now + AIRBORNE_LOCK

                else:
                    # bird: HIGH->NONE / MID->DUCK / LOW->JUMP (your rule)
                    if y2 < head_y:
                        pass  # HIGH->NONE
                    elif y1 < head_y < y2:
                        # MID->DUCK : hold until bird tail passes dino (deadline-based)
                        urgent = (tti is not None and 0 < tti < TTI_DUCK) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                        if urgent and (now - last_duck_t) >= DUCK_COOLDOWN:
                            if speed_px_s is not None:
                                tail_dist = max(0.0, x2 - dino_front_x)
                                hold = tail_dist / speed_px_s + DUCK_PAD
                                hold = clamp(hold, DUCK_HOLD_MIN, DUCK_HOLD_MAX)
                            else:
                                hold = 0.25

                            if not down_held:
                                press_down()
                                down_held = True
                            duck_until = max(duck_until, now + hold)
                            last_duck_t = now
                    else:
                        # LOW->JUMP with emergency override for lock
                        urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                        emergency = (tti is not None and tti < EMERGENCY_TTI) or (dist_hit < EMERGENCY_DIST)
                        if urgent and (now - last_jump_t) >= JUMP_COOLDOWN and (emergency or now >= jump_lock_until):
                            tap_space()
                            last_jump_t = now
                            jump_lock_until = now + AIRBORNE_LOCK

                # write video + show
                vw.write(vis)
                cv2.imshow("Dino Bot (REC)", vis)

                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        finally:
            try:
                if down_held:
                    release_down()
            except Exception:
                pass
            vw.release()
            cv2.destroyAllWindows()

            # print path in terminal (not on video)
            print(f"Saved video: {real_out_path}")

if __name__ == "__main__":
    main()
