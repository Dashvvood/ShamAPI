import requests
import uuid
from rtmlib import draw_skeleton
import cv2
import numpy as np
import asyncio  


async def client(filepath, url="http://127.0.0.1:8000/predict"):
    files = {"file":  open(filepath, "rb")}
    data = {"RequestID": uuid.uuid4().hex}
    response = requests.post(url, data=data, files=files)
    print(response)
    result = response.json()
    img = cv2.imread(filepath)
    keypoints = np.array(result['ModelOutput']['keypoints'])
    scores = np.array(result['ModelOutput']['scores'])
    draw_skeleton(img, keypoints, scores)
    cv2.imwrite(f"processed_{uuid.uuid4().hex}.jpg", img)
    print(f"ProcessedTime: {result['ProcessedTime']}")
    return result

async def main():
    imgs = ["./data/example01.jpg", "./data/example02.jpg"] * 10
    tasks = [client(img) for img in imgs]
    
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())