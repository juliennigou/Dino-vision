import os, glob, random
import cv2
import numpy as np

# adapte si besoin
DATASET_DIR = "dino_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images", "train")
LBL_DIR = os.path.join(DATASET_DIR, "labels", "train")

# d'après ton yaml: 0=cactus, 1=bird, 2=dino
CLS_BIRD = 1
CLS_DINO = 2

def yolo_to_xyxy(line, w, h):
    cls, cx, cy, bw, bh = line.split()
    cls = int(cls)
    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w
    y2 = (cy + bh/2) * h
    return cls, x1, y1, x2, y2

def kmeans_1d(x, k=3, iters=50):
    x = np.asarray(x, dtype=np.float32)
    # init: pick k random points
    c = np.sort(np.random.choice(x, size=k, replace=False))
    for _ in range(iters):
        # assign
        d = np.abs(x[:, None] - c[None, :])
        lab = np.argmin(d, axis=1)
        # update
        new_c = []
        for j in range(k):
            pts = x[lab == j]
            new_c.append(float(np.mean(pts)) if len(pts) else float(c[j]))
        new_c = np.array(new_c, dtype=np.float32)
        new_c.sort()
        if np.allclose(new_c, c, atol=1e-4):
            break
        c = new_c
    return c

def main():
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"Aucune image dans {IMG_DIR}")

    rels = []
    used = 0

    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(LBL_DIR, base + ".txt")
        if not os.path.exists(lp):
            continue

        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lp, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        dinos = []
        birds = []
        for ln in lines:
            cls, x1, y1, x2, y2 = yolo_to_xyxy(ln, w, h)
            if cls == CLS_DINO:
                dinos.append((x1, y1, x2, y2))
            elif cls == CLS_BIRD:
                birds.append((x1, y1, x2, y2))

        if not dinos or not birds:
            continue

        # dino: prendre le plus grand
        d = max(dinos, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        dx1, dy1, dx2, dy2 = d
        dh = max(5.0, (dy2 - dy1))

        for b in birds:
            bx1, by1, bx2, by2 = b
            bcy = (by1 + by2) / 2.0
            rel = (bcy - dy1) / dh
            rels.append(rel)

        used += 1

    rels = np.array(rels, dtype=np.float32)
    if len(rels) < 10:
        raise RuntimeError("Pas assez d'exemples dino+bird dans le dataset.")

    centers = kmeans_1d(rels, k=3)
    # seuils = milieux entre centres triés
    T1 = float((centers[0] + centers[1]) / 2.0)
    T2 = float((centers[1] + centers[2]) / 2.0)

    print("Bird cy_rel stats:")
    print("  n =", len(rels))
    print("  centers (sorted) =", centers.tolist())
    print("Suggested thresholds:")
    print("  T1 (NONE/DUCK) =", T1)
    print("  T2 (DUCK/JUMP) =", T2)
    print("Interpretation:")
    print("  if rel < T1  -> NONE")
    print("  if T1<=rel<T2 -> DUCK")
    print("  if rel >= T2 -> JUMP")

if __name__ == "__main__":
    main()
