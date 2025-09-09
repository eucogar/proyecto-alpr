from typing import List, Tuple
import cv2
import numpy as np
from .config import YOLO_WEIGHTS, CONF_THRESH

try:
    from ultralytics import YOLO  # opcional
except ImportError:
    YOLO = None

   
class PlateDetector:
    def __init__(self):
        self.model = None
        if YOLO and YOLO_WEIGHTS.exists():
            try:
                self.model = YOLO(str(YOLO_WEIGHTS))
            except Exception as e:
                print(f"No se pudo cargar YOLO: {e}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Devuelve lista de boxes [(x1,y1,x2,y2,score), ...]
        Si no hay modelo, retorna el frame completo como ROI para pruebas.
        """
        h, w = frame.shape[:2]
        if self.model is None:
            # MODO MOCK: usar todo el frame como ROI provisional
            return [(0, 0, w, h, 1.0)]

        results = self.model.predict(source=frame, verbose=False)[0]
        boxes = []
        for b in results.boxes:
            conf = float(b.conf.item())
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    @staticmethod
    def crop(frame: np.ndarray, box: Tuple[int, int, int, int, float]) -> np.ndarray:
        x1, y1, x2, y2, _ = box
        x1, y1 = max(0, x1), max(0, y1)
        return frame[y1:y2, x1:x2]
