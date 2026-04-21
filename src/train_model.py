import json
from pathlib import Path

import cv2
import numpy as np

from config import DATASET_DIR, LABELS_FILE, MODEL_FILE, MODELS_DIR


def collect_training_data(dataset_dir: Path):
    faces = []
    labels = []
    label_to_name = {}

    person_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    for label_id, person_dir in enumerate(person_dirs):
        label_to_name[label_id] = person_dir.name

        image_files = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            resized = cv2.resize(img, (200, 200))
            faces.append(resized)
            labels.append(label_id)

    return faces, np.array(labels, dtype=np.int32), label_to_name


def train_model() -> None:
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Dataset directory not found: {DATASET_DIR}")

    faces, labels, label_to_name = collect_training_data(DATASET_DIR)
    if len(faces) == 0:
        raise RuntimeError("No training images found. Capture face samples first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_FILE))

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(label_to_name, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Labels saved to: {LABELS_FILE}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
