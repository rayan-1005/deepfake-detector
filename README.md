# deepfake-detector

A full-stack deepfake video detection system using a 3-head CNN ensemble trained on FaceForensics++, served via a FastAPI backend and visualized through a React forensic dashboard.

---

## Architecture

```
Video Upload (React UI)
        ↓
FastAPI /predict endpoint
        ↓
Frame Sampling → MediaPipe Face Detection
                          ↓
            ┌─────────────────────────┐
            │  Face crop  (224×224)   │ → EfficientNetB0
            │  Eyes crop  (128×128)   │ → MobileNetV2
            │  Nose crop  (128×128)   │ → MobileNetV2
            └─────────────────────────┘
                          ↓
              Ensemble avg → Threshold (0.65) → Verdict
                          ↓
              JSON Response → React Dashboard
```

Three CNN heads independently classify face, eye, and nose regions per frame. Per-frame probabilities are averaged across all three heads, then aggregated across sampled frames. A configurable threshold (default `0.65`) determines the final binary verdict.

---

## Repository Structure

```
deepfake/
├── deepfake-api/                  # FastAPI backend
│   ├── main.py                    # Routes, model loading, inference
│   ├── requirements.txt
│   ├── .gitignore
│   └── models/
│       ├── face_model.keras       # EfficientNetB0 — full face
│       ├── eyes_model.keras       # MobileNetV2 — eye region
│       ├── nose_model.keras       # MobileNetV2 — nose region
│       ├── class_names.json       # ["fake", "real"]
│       └── face_landmarker.task   # Auto-downloaded on first run
│
└── deepfake-ui/                   # React + Vite frontend
    ├── src/
    │   └── App.jsx                # Forensic dashboard UI
    ├── .gitignore
    └── package.json
```

---

## Quickstart

### 1. Backend

```bash
cd deepfake-api

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API available at `http://localhost:8000`.

### 2. Frontend

```bash
cd deepfake-ui
npm install
npm run dev
```

UI available at `http://localhost:5173`.

> Both must be running simultaneously for the full stack to work.

---

## Model Setup

Place the following in `deepfake-api/models/`:

```
face_model.keras
eyes_model.keras
nose_model.keras
class_names.json
```

`face_landmarker.task` is downloaded automatically on first startup if not present.

Models were trained on a subset of [FaceForensics++](https://github.com/ondyari/FaceForensics) (c23 compression, Deepfakes manipulation type). Val accuracy: **90.5%**.

---

## API Reference

### `GET /`
Health check.
```json
{ "status": "ok" }
```

### `POST /predict`
Analyze a video file for deepfake content.

**Request:** `multipart/form-data` with a `file` field (MP4 recommended).

**Response:**
```json
{
  "prediction": "real",
  "fake_probability": 0.3155,
  "real_probability": 0.6845,
  "threshold": 0.65,
  "frames": {
    "tried": 20,
    "faces_found": 20,
    "used": 20
  },
  "per_frame_preds": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

| Field | Description |
|---|---|
| `prediction` | `"fake"` or `"real"` |
| `fake_probability` | Avg probability of being fake (0–1) |
| `real_probability` | Avg probability of being real (0–1) |
| `threshold` | Decision boundary — fake if `fake_probability >= threshold` |
| `frames.tried` | Total frames sampled |
| `frames.faces_found` | Frames with a detected face |
| `frames.used` | Frames used in final prediction |
| `per_frame_preds` | Per-frame binary labels — `0` = fake, `1` = real |

**Error (no face detected):**
```json
{ "error": "No face detected", "faces_found": 0, "tried": 20 }
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/video.mp4"
```

---

## Configuration

Inference parameters in `predict_video()` inside `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `sample_every` | `5` | Sample every N frames |
| `max_tried` | `500` | Max frames to attempt |
| `max_used` | `20` | Max frames used for prediction |
| `threshold` | `0.65` | Fake decision threshold |

Lower `max_used` for faster inference. Raise for higher confidence.

> **Note:** Inference runs on CPU. Expect 20–40s per video. GPU requires a CUDA-enabled TensorFlow build.

---

## Known Limitations

- Degrades on videos with no clearly visible face (occlusion, extreme angles, motion blur)
- Trained on FF++ Deepfakes subset only — may not generalize to Face2Face, FaceSwap, NeuralTextures
- CPU inference is not suitable for real-time or batch workloads without a GPU
- May underperform on heavily compressed or low-resolution inputs

---

## License

For academic and research use only. FaceForensics++ data is subject to its own [Terms of Use](https://github.com/ondyari/FaceForensics/blob/master/dataset/LICENSE).