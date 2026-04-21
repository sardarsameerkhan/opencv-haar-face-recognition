import argparse
import json

import cv2

from config import (
    HAAR_CASCADE_FILE,
    LABELS_FILE,
    MIN_FACE_SIZE,
    MIN_NEIGHBORS,
    MODEL_FILE,
    SCALE_FACTOR,
)


def load_labels():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def recognize_live(camera_index: int = 0, threshold: float = 70.0) -> None:
    if not MODEL_FILE.exists():
        raise RuntimeError(f"Model file not found: {MODEL_FILE}. Run train_model.py first.")
    if not LABELS_FILE.exists():
        raise RuntimeError(f"Labels file not found: {LABELS_FILE}. Run train_model.py first.")

    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_FILE))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_FILE}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check camera permissions and index.")

    print("Starting live recognition. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_resized)
            if confidence <= threshold:
                name = labels.get(label_id, "unknown")
                box_color = (0, 255, 0)
                result = f"{name} ({confidence:.1f})"
            else:
                box_color = (0, 0, 255)
                result = f"unknown ({confidence:.1f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                frame,
                result,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize known faces from webcam feed.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Lower values are stricter. Typical range: 50-80",
    )
    args = parser.parse_args()

    recognize_live(camera_index=args.camera, threshold=args.threshold)


if __name__ == "__main__":
    main()
