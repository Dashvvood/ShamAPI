import requests
from PIL import Image
from elinor.cv import pil_to_b64
from rtmlib import draw_skeleton
import numpy as np

img = Image.open("data/example01.jpg")
img_code = pil_to_b64(img)

message = {
    "image": f"data:image/jpeg;base64,{img_code}"
}
url = "http://127.0.0.1:8000/predict"

response = requests.post(url, json=message)
print(f"{response.status_code = }")
data = response.json()
img = np.array(img)
keypoints = np.array(data["model_output"]["keypoints"])
scores = np.array(data["model_output"]["scores"])
draw = draw_skeleton(img, keypoints, scores, kpt_thr=0.9)
draw = Image.fromarray(draw)

draw.show()

