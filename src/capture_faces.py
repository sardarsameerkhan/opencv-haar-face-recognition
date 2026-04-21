import argparse
from pathlib import Path

import cv2

from config import DATASET_DIR, HAAR_CASCADE_FILE, MIN_FACE_SIZE, MIN_NEIGHBORS, SCALE_FACTOR


def safe_person_name(name: str) -> str:
    cleaned = "_".join(name.strip().split())
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"}).lower()


def capture_faces(person_name: str, num_samples: int = 5, camera_index: int = 0) -> None:
    person_slug = safe_person_name(person_name)
    person_dir = DATASET_DIR / person_slug
    person_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_FILE))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_FILE}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check camera permissions and index.")

    print("\nInstructions:")
    print("1) Position face inside rectangle.")
    print("2) Press SPACE to save detected face sample.")
    print("3) Press Q to quit early.\n")

    captured = 0

    while captured < num_samples:
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

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Person: {person_slug} | Saved: {captured}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Registration (Press SPACE to capture)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:  # Space key
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue

            # Use the largest detected face when multiple are present.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            output_file = person_dir / f"{person_slug}_{captured + 1:02d}.jpg"
            cv2.imwrite(str(output_file), face_resized)
            captured += 1
            print(f"Saved sample {captured}/{num_samples}: {output_file}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Collected {captured} sample(s) for '{person_slug}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face samples using Haar cascade.")
    parser.add_argument("--name", required=True, help="Person name (e.g., sameer)")
    parser.add_argument("--samples", type=int, default=5, help="Number of images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    capture_faces(args.name, args.samples, args.camera)


if __name__ == "__main__":
    main()
