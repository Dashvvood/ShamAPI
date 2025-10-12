import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])
import os
import time
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

from fastapi import FastAPI
from typing import Dict, Any
from rtmlib import Wholebody

from utils.vision_process import fetch_image

import cv2

o_d = elinor.o_d()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

model = Wholebody(**config["model"])
app = FastAPI(title="HPE", version=0.1)

@app.get("/")  
async def index():
    logger.info("Index page accessed")
    return {
        "pid": os.getpid(),
        "config": OmegaConf.to_container(config, resolve=True),
    }

# @app.post("/predict")
# async def predict(file: UploadFile=File(...)):
#     tic = time.time()
#     contents = await file.read()
#     img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
#     keypoints, scores = model(img)
#     toc = time.time()
#     processed_time = round(toc - tic, 4)
#     logger.info(f"ProcessedTime: {processed_time}s")
#     return {
#         "ProcessedTime": processed_time,
#         "ModelOutput": {
#             "keypoints": keypoints.tolist(),
#             "scores": scores.tolist()
#         }
#     }

@app.post("/predict")
async def predict(message: Dict):
    """
    message = {
        "image": "https://example.com/image.jpg"
    }
    """
    tic = time.time()
    image = fetch_image(message)
    img = np.array(image, np.uint8)

    keypoints, scores = model(img)
    toc = time.time()
    processed_time = round(toc - tic, 4)
    logger.info(f"Processed Time: {processed_time}s")
    return {
        "processed_time": processed_time,
        "model_output": {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        }
    }


