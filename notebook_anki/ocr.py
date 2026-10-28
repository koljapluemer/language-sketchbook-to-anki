import sys
from functools import cache

import cv2
import numpy as np
from paddleocr import PaddleOCR

DEVICE = "gpu"


@cache
def _model(lang: str) -> PaddleOCR:
    return PaddleOCR(
        lang=lang,
        device=DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def run(
    img: np.ndarray,
    langs: list[str],
    region: list[int] | None = None,
    mask: list[list[int]] = (),
) -> list[dict]:
    """OCR lines as {id, lang, text, score, bbox} in page coordinates.

    `mask` regions (sketch boxes) are whited out first so drawings don't turn into text.
    Each language's model reads the whole region; overlapping results are expected and
    left for the caller to choose between.
    """
    img = img.copy()
    for x1, y1, x2, y2 in mask:
        img[y1:y2, x1:x2] = 255
    ox, oy = 0, 0
    if region:
        ox, oy, x2, y2 = region
        img = img[oy:y2, ox:x2]

    lines = []
    for lang in langs:
        for res in _model(lang).predict(img):
            for text, score, (x1, y1, x2, y2) in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
                if not text.strip():
                    continue
                lines.append({
                    "lang": lang,
                    "text": text,
                    "score": round(float(score), 2),
                    "bbox": [int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy],
                })
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    for i, line in enumerate(lines, 1):
        line["id"] = f"L{i}"
    return lines


if __name__ == "__main__":
    path, *langs = sys.argv[1:]
    for line in run(cv2.imread(path), langs or ["en"]):
        print(f'{line["id"]:>4} {line["lang"]:>3} {line["score"]:.2f} {line["bbox"]}  {line["text"]}')
