import os, json, tempfile
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ──────────────────────────────────────────
MODELS_DIR = "./models"

print("Loading models...")
face_model = tf.keras.models.load_model(f"{MODELS_DIR}/face_model.keras")
eyes_model = tf.keras.models.load_model(f"{MODELS_DIR}/eyes_model.keras")
nose_model = tf.keras.models.load_model(f"{MODELS_DIR}/nose_model.keras")

with open(f"{MODELS_DIR}/class_names.json") as f:
    class_names = json.load(f)

print("Class names:", class_names)

# ── MediaPipe setup ───────────────────────────────────────
import urllib.request
TASK_PATH = "./models/face_landmarker.task"
if not os.path.exists(TASK_PATH):
    print("Downloading face landmarker...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        TASK_PATH
    )

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpt = mp.tasks.vision.FaceLandmarkerOptions
RunningMode       = mp.tasks.vision.RunningMode

detector = FaceLandmarker.create_from_options(
    FaceLandmarkerOpt(
        base_options=BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )
)

# ── Landmark indices (same as notebook) ──────────────────
LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
NOSE      = [1,2,4,5,6,19,20,94,125,126,209,217,218,237,239,240,241,242]

# ── Crop helpers ─────────────────────────────────────────
def crop_from_landmarks(img, lms, indices, padding=0.35):
    h, w, _ = img.shape
    xs = [int(lms[i].x * w) for i in indices]
    ys = [int(lms[i].y * h) for i in indices]
    x1,x2 = max(0,min(xs)-int((max(xs)-min(xs))*padding)), min(w,max(xs)+int((max(xs)-min(xs))*padding))
    y1,y2 = max(0,min(ys)-int((max(ys)-min(ys))*padding)), min(h,max(ys)+int((max(ys)-min(ys))*padding))
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def crop_face_bbox(img, lms, padding=0.10):
    h, w, _ = img.shape
    xs = [int(p.x * w) for p in lms]
    ys = [int(p.y * h) for p in lms]
    x1,x2 = max(0,min(xs)-int((max(xs)-min(xs))*padding)), min(w,max(xs)+int((max(xs)-min(xs))*padding))
    y1,y2 = max(0,min(ys)-int((max(ys)-min(ys))*padding)), min(h,max(ys)+int((max(ys)-min(ys))*padding))
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def safe_hconcat(a, b, target_h=64):
    def resize_to_h(img, h):
        H,W = img.shape[:2]
        if H<=0 or W<=0: return None
        return cv2.resize(img, (max(1,int(W*(h/H))), h), interpolation=cv2.INTER_AREA)
    a2,b2 = resize_to_h(a,target_h), resize_to_h(b,target_h)
    if a2 is None or b2 is None: return None
    return cv2.hconcat([a2,b2])

def preprocess_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks: return None
    lms = result.face_landmarks[0]
    face = crop_face_bbox(bgr, lms)
    le   = crop_from_landmarks(bgr, lms, LEFT_EYE)
    re   = crop_from_landmarks(bgr, lms, RIGHT_EYE)
    nose = crop_from_landmarks(bgr, lms, NOSE)
    if any(x is None for x in [face, le, re, nose]): return None
    eyes = safe_hconcat(le, re, target_h=64)
    if eyes is None: return None
    face = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
    eyes = cv2.resize(eyes, (128,128), interpolation=cv2.INTER_AREA)
    nose = cv2.resize(nose, (128,128), interpolation=cv2.INTER_AREA)
    return face, eyes, nose

# ── Predict ───────────────────────────────────────────────
def predict_video(path, sample_every=5, max_tried=500, max_used=20, threshold=0.65):
    cap = cv2.VideoCapture(path)
    probs, tried, faces_found, frame_idx = [], 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_every == 0:
            tried += 1
            pack = preprocess_frame(frame)
            if pack:
                faces_found += 1
                face, eyes, nose = pack
                pf = face_model(face[None,...], training=False).numpy()
                pe = eyes_model(eyes[None,...], training=False).numpy()
                pn = nose_model(nose[None,...], training=False).numpy()
                probs.append((pf + pe + pn) / 3)
                if len(probs) >= max_used: break
            if tried >= max_tried: break
        frame_idx += 1
    cap.release()

    if not probs:
        return {"error": "No face detected", "faces_found": 0, "tried": tried}

    probs_arr   = np.vstack(probs)
    avg_prob    = probs_arr.mean(axis=0)
    fake_prob   = float(avg_prob[0])
    pred_idx    = 0 if fake_prob >= threshold else 1
    per_frame   = np.argmax(probs_arr, axis=1).tolist()

    return {
        "prediction":      class_names[pred_idx],
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(float(avg_prob[1]), 4),
        "threshold":        threshold,
        "frames": {
            "tried":       tried,
            "faces_found": faces_found,
            "used":        len(probs),
        },
        "per_frame_preds": per_frame,  # for timeline chart in frontend
    }

# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = predict_video(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result