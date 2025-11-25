from .. import PROJECT_ROOT
from loguru import logger;logger.remove()

import os
import time
import numpy as np
from typing import Dict, Any
from omegaconf import OmegaConf

from rtmlib import Wholebody
from fastapi import FastAPI, Depends
from utils.api_process import get_client_info
from utils.vision_process import fetch_image, fetch_video, fetch_video_generator


config = OmegaConf.load(os.path.join(PROJECT_ROOT, "config/rtmlib.yaml"))
logger.add(**config["log"])

model = Wholebody(**config["model"])
app = FastAPI(title="HPE", version=0.1)
logger.info("Model loaded.")

@app.get("/")  
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"] if "APP_WORKER_ID" in os.environ else None,
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }


@app.post("/image")
async def predict_image(
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
        "api": "/image",
        "model_output": {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        },
        "processed_time": processed_time,
    }


@app.post("/video")
async def predict_video(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
):
    """
    message = {
        "video_url": "https://example.com/video.mp4"
    }
    """
    logger.info(f"Endpoint: /video, IP: {client_info['ip']}; ")

    tic = time.time()
    res = []
    frame_generator, actual_fps, duration = fetch_video_generator(message)
    logger.info(f"Processing video at {actual_fps} fps, duration: {duration:.2f} seconds")

    for i, (frame, timestamp) in enumerate(frame_generator):
        img = np.array(frame, np.uint8)
        keypoints, scores = model(img)
        # Here you can store or process the keypoints and scores as needed
        res.append(
            {
                "frame_idx": i,
                "timestamp": timestamp,
                "keypoints": keypoints.tolist(),
                "scores": scores.tolist()
            }
        )

    toc = time.time()
    processed_time = round(toc - tic, 4)
    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")

    return {
        "api": "/video",
        "fps": actual_fps,
        "duration": duration,
        "num_frames": len(res),
        "model_output": res,
        "processed_time": processed_time,
    }

@app.post("/video/v1")
async def predict_video_v1(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
):
    """
    message = {
        "video_url": "https://example.com/video.mp4"
    }
    """
    logger.info(f"Endpoint: /video, IP: {client_info['ip']}; ")

    tic = time.time()
    res = []
    frames, timestamps, actual_fps, duration = fetch_video(message)
    
    logger.info(f"Processing {len(frames)} frames at {actual_fps} fps, duration: {duration:.2f} seconds")


    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        img = np.array(frame, np.uint8)
        keypoints, scores = model(img)
        # Here you can store or process the keypoints and scores as needed
        res.append(
            {
                "frame_idx": i,
                "timestamp": timestamp,
                "keypoints": keypoints.tolist(),
                "scores": scores.tolist()
            }
        )
        frame_id += 1

    toc = time.time()
    processed_time = round(toc - tic, 4)
    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")

    return {
        "api": "/video",
        "fps": actual_fps,
        "duration": duration,
        "nframes": len(frames),
        "model_output": res,
        "processed_time": processed_time,
    }


# TODO: 流式接口
@app.post("/video-stream")
async def predict_video_stream(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
):
    pass
