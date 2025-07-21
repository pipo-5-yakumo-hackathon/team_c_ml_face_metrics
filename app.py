from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

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

        return {
            "age": int(age),
            "emotions": {k: float(v) for k, v in emotions.items()},
            "skin_redness_rgb": float(r_mean),
            "brightness_l": float(l_mean),
            "red_green_a": float(a_mean),
            "blue_yellow_b": float(b_mean),
            "contrast": float(contrast),
            "hydration": float(hydration)
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)