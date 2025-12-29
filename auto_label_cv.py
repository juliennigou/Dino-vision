import os
import glob
import random
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ---------- CONFIG ----------
RAW_DIR = "dataset_raw"          # dossier des images capturées
OUT_DATASET_DIR = "dataset"      # dataset final (images/labels train/val)
TRAIN_SPLIT = 0.9

# Détection "objets sombres"
THRESH_MODE = "otsu"             # "otsu" ou "fixed"
FIXED_THRESH = 90                # utilisé si THRESH_MODE="fixed" (0..255)

MIN_AREA = 40                    # filtre bruit (aire contour min)
MAX_AREA_FRAC = 0.40             # filtre gros blocs (fraction de l'image)
MORPH_K = 3                      # taille noyau morpho

# Zone du sol: auto-estimation + heuristiques
GROUND_SEARCH_BAND = 0.35        # fraction basse de l'image utilisée pour estimer le sol
CACTUS_NEAR_GROUND_PX = 22       # si bas de box proche du sol -> cactus

# Birds: souvent plus hauts
BIRD_MIN_Y_FROM_TOP_FRAC = 0.10  # ignore tout en haut (nuages/artefacts)
BIRD_MAX_H_FRAC = 0.65           # si box trop haute => probablement pas un bird

# Option: ignorer frames "GAME OVER" (heuristique simple)
FILTER_GAME_OVER = True
GAME_OVER_DARKPIX_FRAC_MAX = 0.22  # si trop de pixels sombres, on soupçonne texte/écran fin

# Random seed
random.seed(0)

# Classes YOLO
CLS_CACTUS = 0
CLS_BIRD = 1


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int

    def as_yolo(self, w: int, h: int) -> str:
        # YOLO: class cx cy bw bh (normalized 0..1)
        cx = (self.x1 + self.x2) / 2.0 / w
        cy = (self.y1 + self.y2) / 2.0 / h
        bw = (self.x2 - self.x1) / w
        bh = (self.y2 - self.y1) / h
        return f"{self.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def ensure_dirs(base: str):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split), exist_ok=True)


def estimate_ground_y(mask: np.ndarray) -> int:
    """
    Estime la ligne du sol via un histogramme de pixels sombres
    en regardant uniquement la bande basse.
    """
    h, w = mask.shape
    band_y0 = int(h * (1.0 - GROUND_SEARCH_BAND))
    band = mask[band_y0:h, :]
    # somme par ligne
    row_sum = band.sum(axis=1) / 255.0  # nb pixels blancs par ligne
    # le sol a souvent une ligne sombre continue -> pic
    idx = int(np.argmax(row_sum))
    ground_y = band_y0 + idx
    return ground_y


def dark_mask(gray: np.ndarray) -> np.ndarray:
    if THRESH_MODE == "otsu":
        # Inverse: objets sombres => blanc
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return m
    else:
        _, m = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY_INV)
        return m


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k = MORPH_K
    kernel = np.ones((k, k), np.uint8)
    # close puis open pour combler + enlever bruit
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m


def maybe_filter_game_over(mask: np.ndarray) -> bool:
    """
    Heuristique : si beaucoup de pixels sombres, il peut y avoir du texte (GAME OVER / score)
    Dans un ROI petit, c'est un proxy simple.
    Retourne True si on DOIT filtrer l'image.
    """
    if not FILTER_GAME_OVER:
        return False
    h, w = mask.shape
    frac = (mask.sum() / 255.0) / (h * w)
    return frac > GAME_OVER_DARKPIX_FRAC_MAX


def find_boxes(mask: np.ndarray, ground_y: int) -> List[Box]:
    h, w = mask.shape
    max_area = int(h * w * MAX_AREA_FRAC)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + bw, y + bh

        # Filtre "trop haut" (nuages)
        if y1 < int(h * BIRD_MIN_Y_FROM_TOP_FRAC) and bh < int(h * 0.12):
            continue

        # Classification cactus vs bird (heuristique)
        bottom_dist = abs(y2 - ground_y)
        near_ground = bottom_dist <= CACTUS_NEAR_GROUND_PX

        if near_ground:
            cls = CLS_CACTUS
        else:
            # bird: plutôt au-dessus du sol et pas une box énorme
            if bh > int(h * BIRD_MAX_H_FRAC):
                continue
            cls = CLS_BIRD

        boxes.append(Box(x1, y1, x2, y2, cls))

    # Option: garder uniquement les 1-2 plus gros objets (souvent un obstacle)
    # Mais on garde tout (cactus groupes) ; tri utile pour debug.
    boxes.sort(key=lambda b: (b.x1, b.y1))
    return boxes


def write_labels(label_path: str, boxes: List[Box], w: int, h: int):
    with open(label_path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(b.as_yolo(w, h) + "\n")


def main():
    ensure_dirs(OUT_DATASET_DIR)

    images = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    if not images:
        raise FileNotFoundError(f"Aucune image .jpg trouvée dans {RAW_DIR}/")

    random.shuffle(images)
    n_train = int(len(images) * TRAIN_SPLIT)

    stats = {
        "total": 0,
        "kept": 0,
        "filtered_game_over": 0,
        "no_boxes": 0,
        "train": 0,
        "val": 0,
    }

    for i, img_path in enumerate(images):
        split = "train" if i < n_train else "val"
        base = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = dark_mask(gray)
        mask = clean_mask(mask)

        stats["total"] += 1

        if maybe_filter_game_over(mask):
            stats["filtered_game_over"] += 1
            continue

        ground_y = estimate_ground_y(mask)
        boxes = find_boxes(mask, ground_y)

        if len(boxes) == 0:
            stats["no_boxes"] += 1
            # On peut quand même sauvegarder image + label vide si tu veux
            # mais pour un dataset d'obstacles, on peut ignorer.
            continue

        out_img = os.path.join(OUT_DATASET_DIR, "images", split, base + ".jpg")
        out_lbl = os.path.join(OUT_DATASET_DIR, "labels", split, base + ".txt")

        shutil.copy2(img_path, out_img)
        write_labels(out_lbl, boxes, w, h)

        stats["kept"] += 1
        stats[split] += 1

    # écrit le yaml
    yaml_path = os.path.join(OUT_DATASET_DIR, "dino.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "path: dataset\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: cactus\n"
            "  1: bird\n"
        )

    print("---- DONE ----")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"YAML: {yaml_path}")
    print("Next: vérifie avec preview_labels.py puis fine-tune YOLO.")


if __name__ == "__main__":
    main()
