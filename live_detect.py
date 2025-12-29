import time
import numpy as np
import cv2
from mss import mss
from ultralytics import YOLO

# ✅ Ta ROI validée
ROI = {"left": 68, "top": 175, "width": 484, "height": 194}
# BGR (OpenCV)
COLORS = {
    "cactus": (0, 255, 0),     # vert
    "bird": (255, 0, 0),       # bleu
    "dino": (0, 255, 255),     # jaune
}


# 🔁 Mets le chemin vers ton best.pt
MODEL_PATH = "runs/detect/train2/weights/best.pt"

CONF = 0.35
SHOW_FPS = True

def main():
    model = YOLO(MODEL_PATH)

    with mss() as sct:
        cv2.namedWindow("Dino Live Detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dino Live Detect", 900, 360)

        last_t = time.time()
        fps = 0.0
        n = 0

        print("Live detect running. Press ESC to quit.")
        print("ROI:", ROI)
        print("Model:", MODEL_PATH)

        while True:
            img = np.array(sct.grab(ROI))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Inference
            res = model.predict(frame, conf=CONF, verbose=False)[0]

            vis = frame.copy()

            # Dessiner boxes + labels
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    cls = int(b.cls[0].cpu().item())
                    conf = float(b.conf[0].cpu().item())

                    name = model.names.get(cls, str(cls))
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                    color = COLORS.get(name, (255, 255, 255))  # fallback blanc

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis,
                        f"{name} {conf:.2f}",
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )


            # FPS
            n += 1
            now = time.time()
            dt = now - last_t
            if dt >= 1.0:
                fps = n / dt
                n = 0
                last_t = now

            if SHOW_FPS:
                cv2.putText(
                    vis,
                    f"FPS: {fps:.1f}  conf={CONF}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Dino Live Detect", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
