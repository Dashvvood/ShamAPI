import base64
from io import BytesIO
import librosa

import audioread
import numpy as np


SAMPLE_RATE = 16000
def process_audio_info(message: dict):
    path = message["audio"]
    if isinstance(path, np.ndarray):
        if path.ndim > 1:
            raise ValueError("Support only mono audio")
        return path
    elif path.startswith("data:audio"):
        _, base64_data = path.split("base64,", 1)
        data = BytesIO(base64.b64decode(base64_data))
    elif path.startswith("http://") or path.startswith("https://"):
        data = audioread.ffdec.FFmpegAudioFile(path)
    elif path.startswith("file://"):
        data = path[len("file://") :]
    else:
        data = path

    return librosa.load(data,sr=SAMPLE_RATE,)[0]
