import cv2
import numpy as np
from typing import List, Tuple

class ColorPlateDetector:
    def __init__(self):
        # Rango ajustado de amarillo para placas colombianas
        self.lower_yellow = np.array([18, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Devuelve lista de boxes [(x1, y1, x2, y2, score)] detectados por color amarillo.
        Con filtros por área, aspecto y posición vertical.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h_img = frame.shape[0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            ratio = w / float(h)

            # Filtros de ruido
            if area < 2000 or area > 50000:
                continue
            if ratio < 1.5 or ratio > 4.5:
                continue
            if y < h_img * 0.3:  # ignorar objetos en la parte superior
                continue

            boxes.append((x, y, x + w, y + h, 1.0))

        return boxes

    @staticmethod
    def crop(frame: np.ndarray, box: Tuple[int, int, int, int, float]) -> np.ndarray:
        x1, y1, x2, y2, _ = box
        return frame[y1:y2, x1:x2]
