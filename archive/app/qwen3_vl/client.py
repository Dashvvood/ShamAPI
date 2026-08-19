from elinor import o_d; o_d = o_d()
import os
import json
import base64
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./data/dummy/video01.mp4", help="Path to input audio file.")
parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/chat", help="URL of the transcription service.")
args = parser.parse_args()


with open(args.input, "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

message = {
    "conversation": [
        {
            "role": "system",
            "content": [
                {"type": "text", 
                 "text": "你是一个有帮助的助手。"
                },
            ],
        },
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": f"{args.input}"},
                {
                    "type": "video", 
                    "video": f"{args.input}",
                    "video_start": 0.0,
                    "video_end": 5.0,
                },
                {"type": "text", "text": "你看到了什么?"},
            ],
        }
    ]
}

response = requests.post(args.url, json=message)

output_path = f"qwen3_vl_{o_d.strftime("%Y%m%d-%H%M%S")}.json"

with open(output_path, "w") as f:
    json.dump(response.json(), f, indent=4, ensure_ascii=False)

print(response.json())
print(response.json().keys())
print(f"Result saved to {output_path}")
