# OpenCV Haar Cascade Face Registration and Recognition

A professional, GitHub-ready Python project for:

1. Registering faces from live webcam images (Haar Cascade + OpenCV).
2. Training a face recognizer (LBPH).
3. Recognizing known faces from live video feed.

## Features

- Portable paths (no hard-coded lab PC paths)
- Haar Cascade face detection (`haarcascade_frontalface_default.xml`)
- LBPH face recognition (`cv2.face.LBPHFaceRecognizer_create`)
- CLI-based workflow for capture, train, and recognize
- Clean project structure for GitHub

## Project Structure

```text
opencv-haar-face-recognition/
|-- data/
|   `-- dataset/
|-- models/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- capture_faces.py
|   |-- train_model.py
|   `-- recognize_live.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.9+
- Webcam
- Windows/Linux/macOS

## Setup

```bash
# 1) Clone or open this folder
cd opencv-haar-face-recognition

# 2) Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

## Step 1: Register Faces (Capture Images)

Capture **5 live photos** per person (you + your neighbor's friend).

```bash
python src/capture_faces.py --name sameer --samples 5
python src/capture_faces.py --name friend --samples 5
```

Controls:

- `SPACE`: save current detected face
- `Q`: quit capture

## Step 2: Train Model

```bash
python src/train_model.py
```

This creates:

- `models/face_lbph_model.yml`
- `models/labels.json`

## Step 3: Recognize From Live Video

```bash
python src/recognize_live.py
```

Optional threshold tuning:

```bash
python src/recognize_live.py --threshold 65
```

## Notes

- Lower threshold = stricter recognition, fewer false positives.
- Keep lighting and camera angle similar between registration and recognition for better results.
- If you use an external camera, pass `--camera 1` (or another index).

## Professional GitHub Push Workflow

```bash
# From project root
cd opencv-haar-face-recognition

# Initialize repo
git init

# Create main branch
git checkout -b main

# First commit
git add .
git commit -m "feat: add OpenCV Haar cascade face registration and live recognition project"

# Connect remote (replace URL)
git remote add origin https://github.com/<your-username>/opencv-haar-face-recognition.git

# Push
git push -u origin main
```

## License

Use for academic/demo purposes. Ensure consent before capturing or recognizing any person's face data.
