import requests
from PIL import Image
from elinor.cv import pil_to_b64

img = Image.open("data/example02.jpg")
img_code = pil_to_b64(img)

message = {
    "image": f"data:image/jpeg;base64,{img_code}"
}
url = "http://127.0.0.1:8000/predict"

response = requests.post(url, json=message)
print(f"{response.status_code = }")
data = response.json()
print(data["model_output"].keys())


