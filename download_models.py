from deepface import DeepFace

models = [
    "VGG-Face",
    "Facenet",
    "ArcFace",
    "Dlib",
    "Age",
    "Gender",
    "Race",
    "Emotion"
]

for model_name in models:
    print(f"Downloading model: {model_name}")
    DeepFace.build_model(model_name)

print("All models downloaded.")