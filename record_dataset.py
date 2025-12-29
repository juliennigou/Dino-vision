import os
import time
import numpy as np
import cv2
from mss import mss

# Mets ici la ROI qui marche (copiée depuis roi_viewer.py)
ROI = {"left": 68, "top": 175, "width": 484, "height": 194}

OUT_DIR = "dataset_raw"
EVERY_N_MS = 80   # 80ms ~ 12.5 fps (bon compromis)
JPEG_QUALITY = 90

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with mss() as sct:
        print("Enregistrement...")
        print("- Appuie sur 's' pour start/stop")
        print("- Appuie sur 'q' pour quitter")
        print(f"ROI: {ROI}")
        print(f"Output: {OUT_DIR}/")

        recording = False
        idx = int(time.time())

        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 1000, 450)

        last = 0.0
        saved = 0

        while True:
            img = np.array(sct.grab(ROI))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Copie UNIQUEMENT pour l'affichage
            preview = frame.copy()

            status = "REC" if recording else "PAUSE"
            cv2.putText(
                preview,
                f"{status} | saved: {saved}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Preview", preview)


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                recording = not recording

            now = time.time()
            if recording and (now - last) * 1000 >= EVERY_N_MS:
                last = now
                filename = os.path.join(OUT_DIR, f"frame_{idx:010d}.jpg")
                idx += 1
                cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                saved += 1

        cv2.destroyAllWindows()
        print("Done. Saved:", saved)

if __name__ == "__main__":
    main()
