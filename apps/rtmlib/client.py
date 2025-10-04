
from urllib import response
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
    result = response.json()
    img = cv2.imread(filepath)
    keypoints = np.array(result['ModelOutput']['keypoints'])
    scores = np.array(result['ModelOutput']['scores'])
    draw_skeleton(img, keypoints, scores)
    return result, img

if __name__ == "__main__":
    result, img = asyncio.run(client("./data/example01.jpg"))
    print(result)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()