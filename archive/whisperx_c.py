from elinor import o_d; o_d = o_d()
import os
import json
import base64
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./data/dummy/audio1.mp3", help="Path to input audio file.")
parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/transcribe", help="URL of the transcription service.")
args = parser.parse_args()

path = args.input
url = args.url

with open(path, "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

message = {
    "audio": "data:audio/mp3;base64," + audio_b64,
    "language": "en",
}

response = requests.post(url, json=message)
with open(f"client_{o_d.strftime("%Y%m%d-%H%M%S")}.json", "w") as f:
    json.dump(response.json(), f, indent=4)
print(response.json())

