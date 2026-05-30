import base64
import io
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# 🎯 内存最高防御：硬限单线程
torch.set_num_threads(1)

app = FastAPI()

# CORS 跨域白名单配置
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

# 全局模型初始化为 None
# 确保 FastAPI 自身安全开机并稳在内存后，再在 startup 事件中请进 AI 模型
model = None

@app.on_event("startup")
def load_model_safely():
    global model
    from ultralytics import YOLO
    print("🚀 App initialized. Loading YOLO weights from local storage...")
    model = YOLO("best.pt")
    # 🎯 刚加载完，立刻地毯式清扫中间产生的临时内存残渣
    gc.collect()
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

        # 极致防爆推理
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

        # 卸磨杀驴，一秒不留
        del img
        del results
        gc.collect()

        return {
            "status": "succeeded",
            "output": output_markers
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
