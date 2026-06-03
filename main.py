import base64
import io
import gc
import torch
import ctypes  # 🎯 Added: For direct communication with the Linux OS memory manager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# 🎯 Force Linux to release empty memory back to the OS immediately
# This prevents Render's 512MB container from automatically restarting after counting
def trim_memory():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("🧹 Linux memory aggressively trimmed and returned to OS.")
    except Exception as e:
        print(f"⚠️ Memory trim skipped: {str(e)}")

# Lock PyTorch to single-thread execution to save processing power
torch.set_num_threads(1)

app = FastAPI()

# CORS security configuration for your GitHub Pages frontend
origins = [
    "https://superlon-ai.github.io",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model_safely():
    global model
    from ultralytics import YOLO
    print("🚀 App initialized. Loading YOLO weights from local storage...")
    model = YOLO("best.pt")
    gc.collect()
    trim_memory()  # Clear memory right after booting up
    print("✅ YOLO Model fully operational inside 512MB container!")

class ImageData(BaseModel):
    image: str

@app.post("/predict")
async def predict_tubes(data: ImageData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")
    try:
        if "," in data.image:
            header, encoded = data.image.split(",", 1)
        else:
            encoded = data.image
        
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 极致防爆推理 (Extreme memory-safe inference)
        with torch.no_grad():
            results = model.predict(img, device="cpu", imgsz=640, conf=0.25)
        
        output_markers = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                output_markers.append({"x": x_center, "y": y_center})

        # 🎯 Clean up memory instantly before the container gets killed
        del img
        del results
        gc.collect()   # Step 1: Tells Python to free the objects
        trim_memory()  # Step 2: Forces Linux to claim the raw RAM back

        return {
            "status": "succeeded",
            "output": output_markers
        }

    except Exception as e:
        gc.collect()
        trim_memory()
        raise HTTPException(status_code=500, detail=str(e))
