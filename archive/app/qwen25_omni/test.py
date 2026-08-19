# -*- coding: utf-8 -*-


import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


#设定单卡分配内存上限max_memory,迫使多卡分片

#环境设置
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,0,3,4,5,")#设置可视卡
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")#
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")#降碎片
# 每张卡的可用配额（按机器调整；3090 每张24GiB，留点余量）
max_mem = {i: "19GiB" for i in range(torch.cuda.device_count())} 
max_mem["cpu"] = "48GiB"  # 兜底


#视频切分
import subprocess,shlex
#把视频裁成很轻量（12s / 10FPS / 336p / 去音轨）
SRC = "XXX.mp4"
CLIP = "YYY.mp4"
subprocess.run(shlex.split(
    f'ffmpeg -y -i "{SRC}" -t 12 -vf "fps=10,scale=-2:336" -an "{CLIP}"'
), check=True)
#"-t 8"表示只截取前八秒，
# fps表示帧率，
# scale=-2：336用来缩放，把高度定为 336，宽度自动按比例调整并取偶数（-2 的含义是“最接近的偶数”，H.264 解码更稳）
# -an：去掉音轨


# use local model
model_dir="ZZZ/Qwen2.5-Omni"
# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto", max_memory = max_mem, offload_folder="/GPFS/data/xingranquan/tmp/offload")
#这里的offload是在设置device_map="auto"且max_memory后，用于临时存放切下来后剩余的权重的。

# 我们建议启用 flash_attention_2 以获取更快的推理速度以及更低的显存占用.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

#禁止语音输出
model.disable_talker()

#定义谈话

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": CLIP},{"type":"text","text":"从这个视频(fps=10)里面，找到有人出现的视频片段，告诉我这些片段的大致时间范围.另外，这个视频总共有多少秒？"}
        ],
    },
]

# set use audio in video  # 该布尔变量决定是否利用视频音轨
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
#数据预处理（text, audio, images, videos）
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
#模型处理
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio 
#模型推理
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
#解码
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text[0])#确认下这里要不要加[0]
#sf.write(
#   "output.wav",
#    audio.reshape(-1).detach().cpu().numpy(),
#    samplerate=24000,
#)

#显示显卡路径
#print("hf_device_map =", getattr(model, "hf_device_map", None))
