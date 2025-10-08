import elinor
DOTENV = elinor.fast_loadenv_then_append_path()

from fastapi import (
    FastAPI, 
    UploadFile, 
    File, Form,
)

import time
import cv2
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from rtmlib import Wholebody
import asyncio

o_d = elinor.o_d()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

model = Wholebody(**config["model"])
app = FastAPI(title="HPE", version=0.1)

@app.get("/")  # 明确设置状态码为200
async def index():
    return {
        "config": OmegaConf.to_container(config, resolve=True)
    }

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    tic = time.time()
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, scores = model(img)
    toc = time.time()
    processed_time = round(toc - tic, 4)
    logger.info(f"ProcessedTime: {processed_time}s")
    return {
        "ProcessedTime": processed_time,
        "ModelOutput": {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        }
    }
