# src/main_alpr.py
import re, cv2
from pathlib import Path
from typing import Dict, Any
from .detect_color import ColorPlateDetector
from .ocr import PlateOCR

PLATE_REGEXES = [
    r"^[A-Z]{3}[0-9]{3}$",
    r"^[A-Z]{3}[0-9]{2}[A-Z]$",
    r"^[A-Z]{2}[0-9]{3}[A-Z]$",
]

def _valid_plate(t: str) -> bool:
    if not t: return False
    s = re.sub(r"[\s\-_]", "", t.upper())
    return any(re.match(p, s) for p in PLATE_REGEXES)

def process_image(img_path: Path, detector: ColorPlateDetector, ocr: PlateOCR) -> Dict[str, Any]:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"image": img_path.name, "error": "no se pudo leer"}

    # detecciones: cada una debe proveer bbox = (x1,y1,x2,y2) y score
    raw = detector.detect(img)
    dets = []
    for r in raw:
        if isinstance(r, dict) and "bbox" in r:
            x1,y1,x2,y2 = map(int, r["bbox"])
            score = float(r.get("score", 1.0))
        elif isinstance(r, (list, tuple)) and len(r) >= 4:
            x1,y1,x2,y2 = map(int, r[:4])
            score = float(r[4]) if len(r) > 4 else 1.0
        else:
            continue

        crop = img[y1:y2, x1:x2]
        try:
            text, conf = ocr.recognize(crop)  # adapta si tu OCR usa otro método
        except Exception:
            text, conf = "", 0.0

        dets.append({
            "bbox": [x1,y1,x2,y2],
            "score": score,
            "plate": text,
            "ocr_conf": float(conf),
            "valid": _valid_plate(text),
        })

    return {"image": img_path.name, "detections": dets}
