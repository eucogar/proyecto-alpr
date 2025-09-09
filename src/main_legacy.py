import cv2
from pathlib import Path
from .detect import PlateDetector
from .ocr import PlateOCR
from .utils import draw_box
from .config import DEFAULT_VIDEO


def run(video_path: str = DEFAULT_VIDEO):
    if not Path(video_path).exists():
        print(f"No se encontró el video en {video_path}. Colócalo y vuelve a ejecutar.")
        return

    cap = cv2.VideoCapture(video_path)
    detector = PlateDetector()
    ocr = PlateOCR(langs=["en"])  # OCR general

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        for b in boxes:
            roi = detector.crop(frame, b)
            plate_text = ocr.read(roi)
            label = plate_text if plate_text else f"score:{b[4]:.2f}"
            draw_box(frame, b, label)

        cv2.imshow("ALPR - demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
