import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:][stats[1:, cv2.CC_STAT_AREA] >= min_area] = 255
    return keep[labels]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # pad so the flood starts outside even when ink touches the image corner
    outside = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = outside.shape
    cv2.floodFill(outside, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(outside[1:-1, 1:-1]))


def _candidates(ink: np.ndarray, unit: int, gap: int) -> list[tuple[float, list[int]]]:
    """Fill every closed outline, then erase everything thin: boxes become solid blocks,
    while handwriting, dots and strokes touching a box from outside disappear."""
    h, w = ink.shape
    closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (unit * gap, unit * gap)))
    filled = _fill_holes(closed)
    min_side = max(w, h) // 18
    solid = cv2.morphologyEx(
        filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_side // 2, min_side // 2))
    )
    boxes = []
    for c in cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_side or bh < min_side:
            continue
        # loose: hand-drawn boxes are skewed and have letters stuck to them
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        rectangularity = cv2.contourArea(c) / max(rw * rh, 1)
        if rectangularity < 0.6:
            continue
        boxes.append((rectangularity, [x, y, x + bw, y + bh]))
    return boxes


def _overlap(a: list[int], b: list[int]) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small


def find_boxes(img: np.ndarray) -> list[list[int]]:
    """Candidate hand-drawn rectangles as [x1, y1, x2, y2], top-to-bottom, left-to-right.

    Tuned for recall: several gap-bridging strengths are combined, and false positives
    (title boxes, boxed labels, letter loops) are expected to be sorted out later.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    unit = max(w, h) // 200  # ~ stroke width

    ink = _drop_specks(_ink_mask(gray), min_area=(unit * 3) ** 2)  # dot grid out
    # extend straight strokes along their own direction so sides that stop short
    # of a corner still meet the neighbouring side
    k = max(w, h) // 60
    for kernel in [(k, 1), (1, k)]:
        line = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        ink = cv2.bitwise_or(ink, cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_RECT, kernel)))

    candidates = [c for gap in (2, 3, 4, 5) for c in _candidates(ink, unit, gap)]
    # the same box shows up in several variants; the most rectangular one is the
    # clean outline, less rectangular ones have neighbouring letters merged in
    candidates.sort(key=lambda c: c[0], reverse=True)
    boxes = []
    for _, c in candidates:
        if all(_overlap(c, b) < 0.5 for b in boxes):
            boxes.append(c)

    boxes.sort(key=lambda b: (b[1] // (h // 20), b[0]))
    return boxes


def crop_sketch(img: np.ndarray, box: list[int]) -> np.ndarray:
    """Crop inside the border, drop the dot grid, white background -> transparent."""
    x1, y1, x2, y2 = box
    inset = max(8, (x2 - x1) // 25)
    crop = img[y1 + inset : y2 - inset, x1 + inset : x2 - inset]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = _ink_mask(gray)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(30, (gray.size // 20000))
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = keep
    return rgba


def annotate(img: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    out = img.copy()
    scale = max(img.shape[:2]) / 1500
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), max(2, int(4 * scale)))
        cv2.putText(out, f"B{i}", (x1 + 5, y1 + int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3 * scale, (0, 0, 255), max(2, int(3 * scale)))
    return out


def extract(image_path: Path, out_dir: Path) -> list[dict]:
    img = cv2.imread(str(image_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = find_boxes(img)
    result = []
    for i, box in enumerate(boxes, 1):
        name = f"B{i}.png"
        cv2.imwrite(str(out_dir / name), crop_sketch(img, box))
        result.append({"id": f"B{i}", "bbox": box, "file": name})
    cv2.imwrite(str(out_dir / "annotated.png"), annotate(img, boxes))
    (out_dir / "boxes.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1])
    for b in extract(path, Path("out") / path.stem):
        print(b["id"], b["bbox"])
