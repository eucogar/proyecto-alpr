import cv2
from pathlib import Path
# from .detect import PlateDetector   # viejo
from .detect_color import ColorPlateDetector

from .ocr import PlateOCR

# Carpeta donde guardaste tus imágenes
IMG_DIR = Path(__file__).resolve().parents[1] / "data" / "samples" / "placas"

def run():
    if not IMG_DIR.exists():
        print(f"No se encontró la carpeta {IMG_DIR}")
        return

    detector = ColorPlateDetector()
    ocr = PlateOCR()

    for img_path in IMG_DIR.glob("*.*"):
        print(f"\nProcesando: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print("No se pudo abrir la imagen.")
            continue

        boxes = detector.detect(img)
        for b in boxes:
            roi = detector.crop(img, b)
            plate_text = ocr.read(roi)
            print("Detectado:", plate_text)

            # Dibujar caja y label en la imagen
            x1, y1, x2, y2, _ = b
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_text:
                cv2.putText(img, plate_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Resultado", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
