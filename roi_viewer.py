import time
import numpy as np
import cv2
from mss import mss

# ROI initiale (à ajuster)
ROI = {"left": 200, "top": 150, "width": 900, "height": 350}

def nothing(_):  # callback pour trackbars
    pass

def clamp_roi(roi, max_w, max_h):
    roi["left"] = int(max(0, min(roi["left"], max_w - 1)))
    roi["top"] = int(max(0, min(roi["top"], max_h - 1)))
    roi["width"] = int(max(1, min(roi["width"], max_w - roi["left"])))
    roi["height"] = int(max(1, min(roi["height"], max_h - roi["top"])))
    return roi

def main():
    with mss() as sct:
        monitor = sct.monitors[1]  # écran principal (souvent 1)
        screen_w, screen_h = monitor["width"], monitor["height"]
        print(f"Screen size (mss): {screen_w}x{screen_h}")
        print("ESC pour quitter")

        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI", 1000, 450)

        # Trackbars pour régler ROI
        cv2.createTrackbar("left", "ROI", ROI["left"], screen_w - 1, nothing)
        cv2.createTrackbar("top", "ROI", ROI["top"], screen_h - 1, nothing)
        cv2.createTrackbar("width", "ROI", ROI["width"], screen_w, nothing)
        cv2.createTrackbar("height", "ROI", ROI["height"], screen_h, nothing)

        fps_t = time.time()
        frames = 0

        while True:
            ROI["left"] = cv2.getTrackbarPos("left", "ROI")
            ROI["top"] = cv2.getTrackbarPos("top", "ROI")
            ROI["width"] = cv2.getTrackbarPos("width", "ROI")
            ROI["height"] = cv2.getTrackbarPos("height", "ROI")
            clamp_roi(ROI, screen_w, screen_h)

            img = np.array(sct.grab(ROI))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # FPS debug
            frames += 1
            if time.time() - fps_t >= 1.0:
                fps = frames / (time.time() - fps_t)
                fps_t = time.time()
                frames = 0
            else:
                fps = None

            if fps is not None:
                cv2.setWindowTitle("ROI", f"ROI (fps ~ {fps:.1f})  {ROI}")

            cv2.imshow("ROI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        print("Final ROI:", ROI)

if __name__ == "__main__":
    main()
