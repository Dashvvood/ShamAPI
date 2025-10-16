import requests
from PIL import Image
from elinor.cv import pil_to_b64
from rtmlib import draw_skeleton
import numpy as np
import aiohttp
import asyncio
from elinor.misc import timer

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
        print(f"{response.status_code = }, {keypoints.shape = }, {scores.shape = }")
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
                    print(f"Status: {response.status}, {keypoints.shape = }, {scores.shape = }")
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
    img = Image.open("data/example01.jpg")
    img_code = pil_to_b64(img)

    message = {
        "image": f"data:image/jpeg;base64,{img_code}"
    }
    url = "http://127.0.0.1:8000/predict"
    messages = [message] * 100

    # 同步请求
    responses = send_request_sync(url, messages)

    # 异步请求
    asyncio.run(send_request_async(url, messages, max_concurrent=10))

