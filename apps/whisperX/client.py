import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])
o_d = elinor.o_d()

import json
import base64
import requests

path = "./data/audio01.mp3"
url  = "http://127.0.0.1:8000/transcribe"
with open(path, "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

message = {
    "audio": "data:audio/mp3;base64," + audio_b64,
    "language": "en"
}

response = requests.post(url, json=message)
with open(f"client_{o_d.strftime("%Y%m%d-%H%M%S")}.json", "w") as f:
    json.dump(response.json(), f, indent=4)
print(response.json())

