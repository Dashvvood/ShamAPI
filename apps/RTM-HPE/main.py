import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])
import os
import time
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
logger.remove()

from fastapi import FastAPI, Depends
from typing import Dict, Any
from rtmlib import Wholebody

from utils.vision_process import fetch_image
from utils.api_process import get_client_info

import cv2

o_d = elinor.o_d()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

model = Wholebody(**config["model"])
app = FastAPI(title="HPE", version=0.1)




@app.get("/")  
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }


@app.post("/predict")
async def predict(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
):
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
    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
    return {
        "processed_time": processed_time,
        "model_output": {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        }   
    }
