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

# Duck short, deadline-based (non-blocking)
DUCK_HOLD_MIN = 0.1
DUCK_HOLD_MAX = 0.12
DUCK_PAD = 0.02
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

def draw_danger_zone(vis, x_start, x_end, alpha=0.16):
    h, w = vis.shape[:2]
    x_start = max(0, min(w - 1, int(x_start)))
    x_end = max(0, min(w, int(x_end)))
    if x_end <= x_start:
        return
    overlay = vis.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, h - 1), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# ---------------- Logger ----------------
class Logger:
    def __init__(self):
        self.last_print = {}
        self.start = time.time()

    def _ts(self):
        return f"{(time.time() - self.start):7.3f}s"

    def log(self, key: str, msg: str, every: float = 0.25):
        """
        Throttle prints: same key printed at most once every `every` seconds.
        """
        now = time.time()
        t = self.last_print.get(key, 0.0)
        if now - t >= every:
            self.last_print[key] = now
            print(f"[{self._ts()}] {msg}")

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
        raise RuntimeError("Classes cactus/bird/dino introuvables dans model.names. Vérifie ton data.yaml.")

    log = Logger()
    log.log("init", f"MODEL={MODEL_PATH} CONF={CONF} ROI={ROI}", every=0)

    last_jump_t = 0.0
    jump_lock_until = 0.0
    last_duck_t = 0.0

    # speed EMA
    prev_cx = None
    prev_t = None
    speed_px_s = None

    # duck deadline
    down_held = False
    duck_until = 0.0

    # target tracking (for logs)
    last_target_sig = None  # (name, rounded_x1, rounded_y1)

    with mss() as sct:
        cv2.namedWindow("Dino Bot (logs)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dino Bot (logs)", 980, 380)

        fps_t = time.time()
        frames = 0
        fps = 0.0

        while True:
            now = time.time()

            # release duck
            if down_held and now >= duck_until:
                release_down()
                down_held = False
                duck_until = 0.0
                log.log("duck_release", "DUCK release (deadline reached)", every=0)

            img = np.array(sct.grab(ROI))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            res = model.predict(frame, conf=CONF, verbose=False)[0]
            boxes = iter_boxes(res)

            vis = frame.copy()

            # dino
            dino_box = pick_best_box_by_conf(boxes, dino_cls)
            if dino_box is None:
                log.log("no_dino", "No dino detected", every=0.5)
                cv2.putText(vis, "No dino", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Dino Bot (logs)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            dx1, dy1, dx2, dy2, _ = dino_box
            dino_front_x = dx2
            dino_h = max(5.0, dy2 - dy1)
            head_y = dy2 - HEAD_FROM_BOTTOM * dino_h

            # obstacles list
            obs = []
            for b in boxes:
                cls = int(b.cls[0].item())
                if cls not in (cactus_cls, bird_cls):
                    continue
                name = names.get(cls, str(cls))
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].item())

                # ahead test using x2 (critical)
                if x2 <= dino_front_x:
                    continue

                lead = LEAD_FRAC_CACTUS if name == NAME_CACTUS else LEAD_FRAC_BIRD
                x_hit = x1 + lead * (x2 - x1)
                dist_hit = x_hit - dino_front_x
                if dist_hit > ROI["width"] * 0.95:
                    continue

                obs.append((name, x1, y1, x2, y2, conf, x_hit, dist_hit, (x1 + x2) / 2.0))

            if not obs:
                log.log("no_obs", "No obstacles ahead", every=0.5)
                cv2.rectangle(vis, (int(dx1), int(dy1)), (int(dx2), int(dy2)), (0, 255, 255), 2)
                cv2.line(vis, (0, int(head_y)), (vis.shape[1] - 1, int(head_y)), (180, 180, 180), 1)
                cv2.imshow("Dino Bot (logs)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # choose target: smallest positive dist_hit
            obs.sort(key=lambda o: (o[7] if o[7] >= 0 else 1e9))
            name, x1, y1, x2, y2, conf, x_hit, dist_hit, cx = obs[0]

            # log target changes (rounded signature)
            sig = (name, int(x1 // 5), int(y1 // 5))
            if sig != last_target_sig:
                last_target_sig = sig
                log.log("target", f"TARGET={name} x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f} dist_hit={dist_hit:.1f}", every=0)

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

            # danger zone visual
            if speed_px_s is not None:
                dz = min(ROI["width"] - 1, int(speed_px_s * TTI_JUMP))
                draw_danger_zone(vis, dino_front_x, dino_front_x + dz)

            action = "NONE"

            # --- decisions + logs ---
            if name == NAME_CACTUS:
                urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_CACTUS)
                if urgent:
                    if now < jump_lock_until:
                        log.log("cactus_block", f"CACTUS urgent but locked (lock_until={jump_lock_until-now:.2f}s)", every=0.2)
                    elif (now - last_jump_t) < JUMP_COOLDOWN:
                        log.log("cactus_cd", f"CACTUS urgent but jump cooldown ({JUMP_COOLDOWN-(now-last_jump_t):.2f}s)", every=0.2)
                    else:
                        tap_space()
                        last_jump_t = now
                        jump_lock_until = now + AIRBORNE_LOCK
                        action = "JUMP"
                        log.log("cactus_jump", f"CACTUS JUMP dist_hit={dist_hit:.1f} tti={tti if tti is not None else None} speed={speed_px_s}", every=0)

            else:
                # bird case by head line
                if y2 < head_y:
                    bird_case = "HIGH->NONE"
                elif y1 < head_y < y2:
                    bird_case = "MID->DUCK"
                else:
                    bird_case = "LOW->JUMP"

                # always log bird case occasionally
                log.log("bird_case", f"BIRD case={bird_case} head_y={head_y:.1f} y1={y1:.1f} y2={y2:.1f} dist_hit={dist_hit:.1f} tti={tti}", every=0.3)

                if bird_case == "MID->DUCK":
                    urgent = (tti is not None and 0 < tti < TTI_DUCK) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                    if urgent:
                        if (now - last_duck_t) < DUCK_COOLDOWN:
                            log.log("duck_cd", f"DUCK urgent but cooldown ({DUCK_COOLDOWN-(now-last_duck_t):.2f}s)", every=0.2)
                        else:
                            hold = DUCK_HOLD_MIN
                            # hold jusqu'à ce que l'arrière du bird (x2) ait passé le dino
                            if speed_px_s is not None:
                                tail_dist = max(0.0, x2 - dino_front_x)  # px
                                hold = tail_dist / speed_px_s + DUCK_PAD
                                hold = clamp(hold, 0.10, 0.45)           # borne large, birds demandent souvent 0.25-0.40s
                            else:
                                hold = 0.25  # fallback si pas de speed

                            if not down_held:
                                press_down()
                                down_held = True

                            duck_until = max(duck_until, now + hold)
                            last_duck_t = now
                            action = f"DUCK({hold:.2f}s)"
                            log.log("duck_do", f"DUCK action hold={hold:.2f}s dist_hit={dist_hit:.1f} tti={tti}", every=0)
                    else:
                        log.log("duck_no", f"DUCK case but not urgent (dist_hit={dist_hit:.1f}, tti={tti})", every=0.4)

                elif bird_case == "LOW->JUMP":
                    urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                    if urgent:
                        if now < jump_lock_until:
                            log.log("bird_jump_block", f"BIRD low urgent but locked ({jump_lock_until-now:.2f}s)", every=0.2)
                        elif (now - last_jump_t) < JUMP_COOLDOWN:
                            log.log("bird_jump_cd", f"BIRD low urgent but jump cooldown ({JUMP_COOLDOWN-(now-last_jump_t):.2f}s)", every=0.2)
                        else:
                            tap_space()
                            last_jump_t = now
                            jump_lock_until = now + AIRBORNE_LOCK
                            action = "JUMP"
                            log.log("bird_jump", f"BIRD low JUMP dist_hit={dist_hit:.1f} tti={tti}", every=0)
                    else:
                        log.log("bird_jump_no", f"BIRD low but not urgent (dist_hit={dist_hit:.1f}, tti={tti})", every=0.4)

            # ---- Debug draw ----
            cv2.rectangle(vis, (int(dx1), int(dy1)), (int(dx2), int(dy2)), (0, 255, 255), 2)
            cv2.line(vis, (0, int(head_y)), (vis.shape[1] - 1, int(head_y)), (180, 180, 180), 1)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            cv2.circle(vis, (int(x_hit), int((y1 + y2) / 2)), 4, (255, 255, 255), -1)

            speed_txt = "None" if speed_px_s is None else f"{speed_px_s:.0f}"
            tti_txt = "None" if tti is None else f"{tti:.2f}"
            cv2.putText(vis, f"action={action} dist={dist_hit:.0f} tti={tti_txt} speed={speed_txt}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # FPS
            frames += 1
            if time.time() - fps_t >= 1.0:
                fps = frames / (time.time() - fps_t)
                fps_t = time.time()
                frames = 0

            cv2.putText(vis, f"FPS {fps:.1f} downHeld={down_held}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Dino Bot (logs)", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        try:
            if down_held:
                release_down()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
