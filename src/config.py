from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "samples"
MODELS_DIR = ROOT / "models"


# Ruta por defecto del video de pruebas
DEFAULT_VIDEO = str(DATA_DIR / "video.mp4")


# Regex básica para placa colombiana: 3 letras + 3 dígitos (autos). Ajusta para motos si lo necesitas
PLATE_REGEX = re.compile(r"^[A-Z]{3}[0-9]{3}$")


# Score mínimo de confianza para detección (si usas YOLO)
CONF_THRESH = 0.4


# Si no tienes un detector de placas aún, deja None. Cuando tengas pesos, apunta aquí.
YOLO_WEIGHTS = MODELS_DIR / "yolov8lpr.pt" # reemplaza por tus pesos reales