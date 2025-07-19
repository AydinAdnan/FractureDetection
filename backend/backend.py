from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import shutil
import os
from ultralytics import YOLO
import uuid

app = FastAPI()

# Allow frontend (adjust origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained fracture model if needed

# CLAHE function
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for YOLO

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image to disk
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply CLAHE
    img_clahe = apply_clahe(img)

    # Run YOLOv8 prediction
    results = model(img_clahe)

    # Visualize and save output
    result_img = results[0].plot()
    filename = f"pred_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)
    cv2.imwrite(output_path, result_img)

    # Send back image URL
    return JSONResponse(content={"result_image_url": f"/static/{filename}"})

@app.get("/static/{filename}")
async def get_image(filename: str):
    file_path = os.path.join("static", filename)
    return FileResponse(file_path)
