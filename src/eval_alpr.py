from .detect_color import ColorPlateDetector
from .ocr import PlateOCR
from .main_alpr import process_image, PLATE_REGEXES
from pathlib import Path
import json, argparse, re

def norm(s): 
    return re.sub(r"[\s\-_]", "", s.upper())

def load_gt(p): 
    items=json.load(open(p,"r",encoding="utf-8"))
    return {it["image"]:{norm(x) for x in it["plates"]} for it in items}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--imgs","-i",default="data/samples/placas")
    ap.add_argument("--gt","-g",default="data/gt/labels.json")
    args=ap.parse_args()

    gt=load_gt(args.g)
    det, ocr = ColorPlateDetector(), PlateOCR()
    tp=fp=fn=0
    for imgp in sorted(Path(args.imgs).iterdir()):
        if imgp.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}: 
            continue
        res=process_image(imgp, det, ocr)
        pred={norm(d["plate"]) for d in res.get("detections",[]) if d.get("plate") and any(re.match(p, norm(d["plate"])) for p in PLATE_REGEXES)}
        truth=gt.get(imgp.name,set())
        tp+=len(pred & truth); fp+=len(pred - truth); fn+=len(truth - pred)
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec =tp/(tp+fn) if tp+fn else 0.0
    f1 =2*prec*rec/(prec+rec) if prec+rec else 0.0
    print(f"TP:{tp} FP:{fp} FN:{fn} | Precision:{prec:.3f} Recall:{rec:.3f} F1:{f1:.3f}")

if __name__=="__main__":
    main()
