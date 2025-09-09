import cv2
from pathlib import Path
from .detect import PlateDetector
from .ocr import PlateOCR

IMG_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "car.jpg"

def run():
    if not IMG_PATH.exists():
        print(f"No se encontró la imagen {IMG_PATH}")
        return
    img = cv2.imread(str(IMG_PATH))
    detector = PlateDetector()
    ocr = PlateOCR()

    boxes = detector.detect(img)
    for b in boxes:
        roi = detector.crop(img, b)
        plate_text = ocr.read(roi)
        print("Detectado:", plate_text)

    cv2.imshow("Test Image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    run()
