from .. import PROJECT_ROOT
from elinor import o_d; o_d = o_d()

import requests
from PIL import Image
from elinor.cv import pil_to_b64
from rtmlib import draw_skeleton
import numpy as np
import aiohttp
import asyncio
from elinor.misc import timer
import argparse


@timer
def send_request_sync(url, messages,):
    """
    message = {
        "image": f"data:image/jpeg;base64,{img_code}"
    }
    """
    res = []
    for message in messages:
        response = requests.post(url, json=message)
        data = response.json()
        keypoints = np.array(data["model_output"]["keypoints"])
        scores = np.array(data["model_output"]["scores"])
        res.append(data)
    return res

@timer
async def send_request_async(url, messages, max_concurrent=10):
    """
    简化的异步版本
    max_concurrent: 最大并发数，避免服务器压力过大
    """
    res = []
    
    # 限制并发数的信号量
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def single_request(message):
        async with semaphore:  # 控制并发
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=message) as response:
                    data = await response.json()
                    keypoints = np.array(data["model_output"]["keypoints"])
                    scores = np.array(data["model_output"]["scores"])
                    return data
    
    # 创建所有任务
    tasks = [single_request(message) for message in messages]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for result in results:
        if not isinstance(result, Exception):
            res.append(result)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./data/dummy/image01.jpg", help="Path to input audio file.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/image", help="URL of the transcription service.")
    args = parser.parse_args()

    img = Image.open(args.input)
    img_code = pil_to_b64(img)

    message = {
        "image": f"data:image/jpeg;base64,{img_code}"
    }
    messages = [message] * 10

    # 同步请求
    responses = send_request_sync(args.url, messages)
    for data in responses[:3]:
        keypoints = np.array(data["model_output"]["keypoints"])
        scores = np.array(data["model_output"]["scores"])
        print(f"{keypoints.shape = }, {scores.shape = }")

    
    # 异步请求
    responses = asyncio.run(send_request_async(args.url, messages, max_concurrent=10))
    for data in responses[:3]:
        keypoints = np.array(data["model_output"]["keypoints"])
        scores = np.array(data["model_output"]["scores"])
        print(f"{keypoints.shape = }, {scores.shape = }")
    