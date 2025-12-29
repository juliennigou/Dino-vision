import time
import numpy as np
import cv2
from mss import mss
from ultralytics import YOLO
from pynput.keyboard import Controller, Key

# ---------------- Config ----------------
ROI = {"left": 68, "top": 175, "width": 484, "height": 194}
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF = 0.35

# Timing (TTI = dist_hit / speed)
TTI_JUMP = 0.33
TTI_DUCK = 0.26
MIN_SPEED = 60.0
MAX_TTI = 1.2

# Fallback pixel triggers (si speed/tti pas dispo)
PIX_TRIGGER_CACTUS = 135
PIX_TRIGGER_BIRD   = 165  # birds souvent besoin + marge

# Anti-spam jump
JUMP_COOLDOWN = 0.22
AIRBORNE_LOCK = 0.28

# Duck hold non-bloquant et court (deadline)
DUCK_HOLD_MIN = 0.05
DUCK_HOLD_MAX = 0.12
DUCK_PAD      = 0.02
DUCK_COOLDOWN = 0.16

# Point d'impact en X (comme cactus)
LEAD_FRAC_CACTUS = 0.72
LEAD_FRAC_BIRD   = 0.20

# ⭐ Ligne "tête du dino" calculée depuis le bas (plus stable)
# head_y = dy2 - HEAD_FROM_BOTTOM * dino_h
# ex: 0.78 => tête assez haute (vers le haut de la box)
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

# ---------------- Main ----------------
def main():
    model = YOLO(MODEL_PATH)
    names = model.names

    dino_cls = cactus_cls = bird_cls = None
    for k, v in names.items():
        if v == NAME_DINO: dino_cls = k
        elif v == NAME_CACTUS: cactus_cls = k
        elif v == NAME_BIRD: bird_cls = k
    if None in (dino_cls, cactus_cls, bird_cls):
        raise RuntimeError("Classes cactus/bird/dino introuvables dans model.names. Vérifie ton data.yaml.")

    last_jump_t = 0.0
    jump_lock_until = 0.0
    last_duck_t = 0.0

    # vitesse: EMA sur centre de la cible choisie
    prev_cx = None
    prev_t = None
    speed_px_s = None

    # duck state (deadline)
    down_held = False
    duck_until = 0.0

    with mss() as sct:
        cv2.namedWindow("Dino Bot (bird fixed)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dino Bot (bird fixed)", 980, 380)

        fps_t = time.time()
        frames = 0
        fps = 0.0

        print("Running. ESC to quit.")

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

            vis = frame.copy()
            action = "NONE"
            debug_line = ""

            # Dino
            dino_box = pick_best_box_by_conf(boxes, dino_cls)
            if dino_box is None:
                cv2.putText(vis, "No dino detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Dino Bot (bird fixed)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            dx1, dy1, dx2, dy2, _ = dino_box
            dino_front_x = dx2
            dino_h = max(5.0, dy2 - dy1)
            head_y = dy2 - HEAD_FROM_BOTTOM * dino_h

            # Collect obstacles with dist_hit (IMPORTANT: use x2 test, not x1)
            obs = []  # list of dict: {name,x1,y1,x2,y2,conf,x_hit,dist_hit,cx}
            for b in boxes:
                cls = int(b.cls[0].item())
                if cls not in (cactus_cls, bird_cls):
                    continue
                name = names.get(cls, str(cls))
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].item())

                # "ahead" criterion: not fully behind dino
                if x2 <= dino_front_x:
                    continue

                lead = LEAD_FRAC_CACTUS if name == NAME_CACTUS else LEAD_FRAC_BIRD
                x_hit = x1 + lead * (x2 - x1)
                dist_hit = x_hit - dino_front_x

                # ignore if ridiculously far (keeps speed stable)
                if dist_hit > ROI["width"] * 0.95:
                    continue

                obs.append({
                    "name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf,
                    "x_hit": x_hit, "dist_hit": dist_hit, "cx": (x1 + x2) / 2.0
                })

            if not obs:
                # nothing to do
                cv2.rectangle(vis, (int(dx1), int(dy1)), (int(dx2), int(dy2)), (0,255,255), 2)
                cv2.line(vis, (0, int(head_y)), (vis.shape[1]-1, int(head_y)), (180,180,180), 1)
                cv2.imshow("Dino Bot (bird fixed)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # Choose target by smallest positive dist_hit (or smallest absolute if negative small)
            obs.sort(key=lambda o: (o["dist_hit"] if o["dist_hit"] >= 0 else 1e9))
            target = obs[0]

            # Update speed EMA from target center (do NOT reset when target changes)
            if prev_cx is not None and prev_t is not None:
                dt = now - prev_t
                if dt > 1e-3:
                    v = (prev_cx - target["cx"]) / dt  # px/s, should be >0
                    if v > MIN_SPEED:
                        speed_px_s = v if speed_px_s is None else 0.75 * speed_px_s + 0.25 * v
            prev_cx = target["cx"]
            prev_t = now

            # Compute TTI
            dist_hit = target["dist_hit"]
            tti = None
            if speed_px_s is not None and dist_hit > 0:
                tti = dist_hit / speed_px_s

            # Danger zone visual
            if speed_px_s is not None:
                dz = min(ROI["width"] - 1, int(speed_px_s * TTI_JUMP))
                draw_danger_zone(vis, dino_front_x, dino_front_x + dz)

            # Decide "urgent" (TTI OR pixel fallback)
            def urgent_for(tt, pix):
                if tt is not None:
                    return 0 < tt < MAX_TTI and tt < (TTI_DUCK if pix == PIX_TRIGGER_BIRD else TTI_JUMP)
                else:
                    return dist_hit < pix

            # ---- ACTION LOGIC ----
            if target["name"] == NAME_CACTUS:
                urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_CACTUS)
                if urgent and now >= jump_lock_until and (now - last_jump_t) >= JUMP_COOLDOWN:
                    tap_space()
                    last_jump_t = now
                    jump_lock_until = now + AIRBORNE_LOCK
                    action = "JUMP"
            else:
                # ✅ TA RÈGLE bird (3 cas) avec head_y
                by1, by2 = target["y1"], target["y2"]

                # case A: bird completely above head -> NONE
                if by2 < head_y:
                    bird_case = "HIGH->NONE"
                    action = "NONE"
                # case B: bird crosses head line -> DUCK
                elif by1 < head_y < by2:
                    bird_case = "MID->DUCK"
                    urgent = (tti is not None and 0 < tti < TTI_DUCK) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                    if urgent and (now - last_duck_t) >= DUCK_COOLDOWN:
                        # hold very short, deadline-based (non-bloquant)
                        hold = DUCK_HOLD_MIN
                        if speed_px_s is not None:
                            bw = max(1.0, target["x2"] - target["x1"])
                            hold = bw / speed_px_s + DUCK_PAD
                            hold = clamp(hold, DUCK_HOLD_MIN, DUCK_HOLD_MAX)

                        if not down_held:
                            press_down()
                            down_held = True
                        duck_until = max(duck_until, now + hold)
                        last_duck_t = now
                        action = f"DUCK({hold:.2f}s)"
                # case C: bird completely below head -> JUMP
                else:
                    bird_case = "LOW->JUMP"
                    urgent = (tti is not None and 0 < tti < TTI_JUMP) or (tti is None and dist_hit < PIX_TRIGGER_BIRD)
                    if urgent and now >= jump_lock_until and (now - last_jump_t) >= JUMP_COOLDOWN:
                        tap_space()
                        last_jump_t = now
                        jump_lock_until = now + AIRBORNE_LOCK
                        action = "JUMP"

                debug_line = f"{bird_case} by1={by1:.0f} by2={by2:.0f} head={head_y:.0f}"

            # ---- Debug draw ----
            cv2.rectangle(vis, (int(dx1), int(dy1)), (int(dx2), int(dy2)), (0,255,255), 2)
            cv2.line(vis, (0, int(head_y)), (vis.shape[1]-1, int(head_y)), (180,180,180), 1)

            tx1, ty1, tx2, ty2 = map(int, (target["x1"], target["y1"], target["x2"], target["y2"]))
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255,255,255), 2)
            cv2.circle(vis, (int(target["x_hit"]), int((target["y1"] + target["y2"]) / 2)), 4, (255,255,255), -1)

            speed_txt = "None" if speed_px_s is None else f"{speed_px_s:.0f}"
            tti_txt = "None" if tti is None else f"{tti:.2f}"
            cv2.putText(vis, f"action={action} dist={dist_hit:.0f} tti={tti_txt} speed={speed_txt}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if debug_line:
                cv2.putText(vis, debug_line, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            frames += 1
            if time.time() - fps_t >= 1.0:
                fps = frames / (time.time() - fps_t)
                fps_t = time.time()
                frames = 0

            cv2.putText(vis, f"FPS {fps:.1f} downHeld={down_held}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Dino Bot (bird fixed)", vis)
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
