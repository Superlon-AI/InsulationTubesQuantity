import base64
import io
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

# 🎯 内存优化 1：强行限制 PyTorch 线程数为 1
# Render 512MB 方案只有 1 核 CPU，限制为 1 线程能死死压住多线程并发带来的运行内存暴增
torch.set_num_threads(1)

app = FastAPI()

# 🎯 CORS 跨域配置：已为你精准配置好 Superlon-AI 的 GitHub Pages 专属域名
origins = [
    "https://superlon-ai.github.io",
    "http://localhost:3000",  # 留作本地测试使用
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎯 内存优化 2：在程序启动时，一次性将轻量化模型（约 6.2MB）加载进内存
# 严禁将加载模型的代码写在 predict 函数内部，否则每传一张图就会叠加一次内存，两张图就会爆内存！
model = YOLO("best.pt")

class ImageData(BaseModel):
    image: str  # 接收前端传来的图片 Base64 字符串

@app.post("/predict")
async def predict_tubes(data: ImageData):
    try:
        # 1. 解析并解码前端发来的 Base64 图片数据
        if "," in data.image:
            header, encoded = data.image.split(",", 1)
        else:
            encoded = data.image
        
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. 🎯 内存优化 3：彻底封印梯度内存消耗
        # force 使用 CPU 运行，imgs=640 匹配前端压缩尺寸，with torch.no_grad() 禁掉所有中间计算缓存
        with torch.no_grad():
            results = model.predict(img, device="cpu", imgsz=640, conf=0.25)
        
        # 3. 提取保温管的中心点坐标 (x, y)，完美对接你原先 React 前端的渲染画布
        output_markers = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # 获取 [左上x, 左上y, 右下x, 右下y]
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                output_markers.append({"x": x_center, "y": y_center})

        # 4. 🎯 内存优化 4：卸磨杀驴，干完活立刻销毁图片和结果大变量
        del img
        del results
        
        # 5. 强行触发 Python 垃圾回收机制，人肉清空运行内存残渣
        gc.collect()

        return {
            "status": "succeeded",
            "output": output_markers
        }

    except Exception as e:
        # 哪怕中途发生未知错误，也必须确保清理一次内存，防止死锁爆内存
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
