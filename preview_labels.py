import os
import glob
import random
import cv2

DATASET_DIR = "dataset"
SPLIT = "train"      # "train" ou "val"
N_SAMPLES = 80

# classes
NAMES = {0: "cactus", 1: "bird"}

def yolo_to_xyxy(line, w, h):
    cls, cx, cy, bw, bh = line.strip().split()
    cls = int(cls)
    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return cls, x1, y1, x2, y2

def main():
    img_dir = os.path.join(DATASET_DIR, "images", SPLIT)
    lbl_dir = os.path.join(DATASET_DIR, "labels", SPLIT)

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    random.shuffle(imgs)
    imgs = imgs[:N_SAMPLES]

    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, base + ".txt")

        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                lines = [x for x in f.read().splitlines() if x.strip()]
        else:
            lines = []

        vis = img.copy()
        for line in lines:
            cls, x1, y1, x2, y2 = yolo_to_xyxy(line, w, h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(vis, NAMES.get(cls, str(cls)), (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("preview", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
