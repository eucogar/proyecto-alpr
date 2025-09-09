import re
import cv2

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR no instalado. Ejecuta: pip install easyocr")

class PlateOCR:
    def __init__(self):
        if OCR_AVAILABLE:
            # Usamos idioma inglés porque maneja letras/números latinos
            self.reader = easyocr.Reader(["en"])
        else:
            self.reader = None

    def preprocess(self, roi):
        """Preprocesa la imagen para mejorar OCR"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Ecualizar histograma para mejorar contraste
        gray = cv2.equalizeHist(gray)
        # Umbral adaptativo para resaltar caracteres
        proc = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return proc

    def correct_text(self, text: str) -> str:
        """Corrige errores comunes de OCR"""
        corrections = {"O": "0", "I": "1", "S": "5"}
        for k, v in corrections.items():
            text = text.replace(k, v)
        return text

    def read(self, roi):
        if not self.reader:
            return None

        # 🔹 Recortar la zona central de la placa
        # Para evitar que OCR lea "COLOMBIA" en la parte inferior
        h, w = roi.shape[:2]
        roi_focus = roi[int(h*0.2):int(h*0.7), :]

        candidates = []
        # OCR directo
        candidates.extend(self.reader.readtext(roi_focus, detail=0))
        # OCR con preprocesamiento
        roi_proc = self.preprocess(roi_focus)
        candidates.extend(self.reader.readtext(roi_proc, detail=0))

        if not candidates:
            return None

        # 🔹 Revisar todos los resultados
        for cand in candidates:
            text = cand.upper().replace(" ", "")
            text = "".join(ch for ch in text if ch.isalnum())
            text = self.correct_text(text)

            # Validar patrón de placa colombiana AAA111
            match = re.findall(r"[A-Z]{3}[0-9]{3}", text)
            if match:
                return match[0]

        # Si nada cumple el patrón, devolvemos None
        return None
