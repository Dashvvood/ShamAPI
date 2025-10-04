import os
import elinor
# 环境变量
DOTENV = elinor.fast_loadenv_then_append_path()
from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
from rtmlib import Wholebody
from omegaconf import OmegaConf
from loguru import logger
import uvicorn

o_d = elinor.o_d()
config = OmegaConf.load("./config/default.yaml")
logger.add(**config["log"])
logger.info(f"Config: {config}")


model = Wholebody(**config["model"])
app = FastAPI(title="HPE", version=0.1)

@app.get("/")
async def index():
    return config

@app.post("/predict")
async def predict(RequestID: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"RequestID: {RequestID}, filename: {file.filename}, content_type: {file.content_type}")
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, scores = model(img)
    return {
        "RequestID": RequestID,
        "ModelOutput": {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config["app"]["host"], 
        port=config["app"]["port"],
    )
