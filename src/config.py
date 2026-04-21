from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "face_lbph_model.yml"
LABELS_FILE = MODELS_DIR / "labels.json"

# OpenCV includes Haar cascade files; this path is portable across machines.
HAAR_CASCADE_FILE = Path(__import__("cv2").data.haarcascades) / "haarcascade_frontalface_default.xml"

# Detection parameters
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (80, 80)
