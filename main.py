import os
import urllib.request
import base64
import io
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

# 🎯 512MB 内存压榨优化：限制 PyTorch 线程数为 1
torch.set_num_threads(1)

app = FastAPI()

# 🎯 CORS 跨域治理：允许来自你 GitHub Pages 的访问
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

# 🎯 GitHub 云端极速空投机制
MODEL_PATH = "best.pt"
# 这里已经完美换成了你刚才抓到的对头链接！
MODEL_URL = "https://github.com/Superlon-AI/InsulationTubesCount/releases/download/v1.0/best.pt"

if not os.path.exists(MODEL_PATH):
    print("🚀 Detecting no best.pt in Render storage. Pulling from GitHub Cloud Releases...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Weights fully loaded and saved locally!")
    except Exception as e:
        print(f"❌ Failed to download weights: {str(e)}")

# 加载刚空投成功的本地权重模型（6.2MB 核心加载）
model = YOLO(MODEL_PATH)

class ImageData(BaseModel):
    image: str

@app.post("/predict")
async def predict_tubes(data: ImageData):
    try:
        # 1. 解码前端发来的 Base64 图片
        if "," in data.image:
            header, encoded = data.image.split(",", 1)
        else:
            encoded = data.image
        
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. 🎯 强制单核推理，彻底关闭梯度缓存
        with torch.no_grad():
            results = model.predict(img, device="cpu", imgsz=640, conf=0.25)
        
        # 3. 计算管子中心点，完美契合 React 前端渲染
        output_markers = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                output_markers.append({"x": x_center, "y": y_center})

        # 4. 🎯 人肉强行回收运行内存残渣，严防 OOM 闪退
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
