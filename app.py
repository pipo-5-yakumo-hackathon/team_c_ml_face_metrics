from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import math

class MaxSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int = 1_000_000_000):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_upload_size:
            return JSONResponse(content={"error": "File too large"}, status_code=413)
        return await call_next(request)

app = FastAPI()
app.add_middleware(MaxSizeLimitMiddleware, max_upload_size=1_000_000_000)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def hydration_score(hydration, mu=200, sigma_left=150, sigma_right=295):
    if hydration <= mu:
        score = 100 * math.exp(-0.5 * ((hydration - mu) / sigma_left) ** 2)
    else:
        score = 100 * math.exp(-0.5 * ((hydration - mu) / sigma_right) ** 2)
    return max(0, min(100, score))

def redness_score(redness, x0=140, k=0.05):
    x = redness
    score = 100 - (40 / (1 + math.exp(-k * (x - x0))))
    if x > 180:
        score = max(0, 60 - (x - 180) * 0.8)
    return max(0, min(100, score))

def extract_skin_mask(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for idx in range(234, 454):
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                landmarks.append([x, y])
            hull = cv2.convexHull(np.array(landmarks))
            cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask

def evaluate_metrics(age, r_mean, l_mean, a_mean, b_mean, contrast, hydration):
    return {
        "age": {
            "value": int(age),
            "status": "ok"
        },
        "skin_redness_rgb": {
            "value": round(r_mean, 1),
            "status": "high (possible irritation)" if r_mean > 180 else "normal" if r_mean > 100 else "low"
        },
        "brightness_l": {
            "value": round(l_mean, 1),
            "status": "light" if l_mean > 70 else "normal" if l_mean > 50 else "dark"
        },
        "red_green_a": {
            "value": round(a_mean, 1),
            "status": "redness" if a_mean > 20 else "normal" if a_mean > 5 else "pale"
        },
        "blue_yellow_b": {
            "value": round(b_mean, 1),
            "status": "yellow tint" if b_mean > 30 else "normal" if b_mean > 10 else "cool"
        },
        "contrast": {
            "value": round(contrast, 1),
            "status": "low (fatigue)" if contrast < 35 else "normal" if contrast <= 70 else "high (harsh light)"
        },
        "hydration": {
            "value": round(hydration, 1),
            "status": "dry" if hydration < 100 else "normal" if hydration <= 300 else "well-hydrated (over-hydrated)"
        }
    }

@app.on_event("startup")
async def startup_event():
    global face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    _ = DeepFace.analyze(
        img_path=np.zeros((100, 100, 3), dtype=np.uint8),
        actions=['emotion', 'age'],
        enforce_detection=False
    )

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        result = DeepFace.analyze(img_path=img, actions=['emotion', 'age'], enforce_detection=False)
        emotions = result[0]['emotion']
        age = result[0]['age']

        mask = extract_skin_mask(img)
        if np.sum(mask) == 0:
            return JSONResponse(content={"error": "No face detected"}, status_code=400)

        skin_rgb = img[mask > 0]
        r_mean = float(np.mean(skin_rgb[:, 2]) if skin_rgb.size > 0 else 0)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = lab[:, :, 0] * (100 / 255)
        lab[:, :, 1] = lab[:, :, 1] - 128
        lab[:, :, 2] = lab[:, :, 2] - 128
        skin_lab = lab[mask > 0]

        l_mean = float(np.mean(skin_lab[:, 0]) if skin_lab.size > 0 else 0)
        a_mean = float(np.mean(skin_lab[:, 1]) if skin_lab.size > 0 else 0)
        b_mean = float(np.mean(skin_lab[:, 2]) if skin_lab.size > 0 else 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        hydration = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        metrics =  {
            "age": int(age),
            "emotions": {k: float(v) for k, v in emotions.items()},
            "skin_redness_rgb": float(r_mean),
            "brightness_l": float(l_mean),
            "red_green_a": float(a_mean),
            "blue_yellow_b": float(b_mean),
            "contrast": float(contrast),
            "hydration": float(hydration)
        }

        evaluated = evaluate_metrics(
            metrics["age"],
            metrics["skin_redness_rgb"], 
            metrics["brightness_l"], 
            metrics["red_green_a"],
            metrics["blue_yellow_b"],
            metrics["contrast"],
            metrics["hydration"]
        )

        r_score = redness_score(metrics["skin_redness_rgb"])
        h_score = hydration_score(metrics["hydration"])
            
        overall_score = (r_score + h_score) / 2
        overall_score =  round(overall_score, 1)

        evaluated["skin_redness_rgb"]["value"] = r_score
        evaluated["hydration"]["value"]        = h_score

        return {
            "emotions"   : metrics["emotions"],
            "metrics"    : evaluated,
            "skin_score" : overall_score
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)