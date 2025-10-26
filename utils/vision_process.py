# Copyright https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
import base64
import copy
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import requests
import os
import tempfile
import numpy as np
from fastapi import HTTPException
from loguru import logger
from PIL import Image
from moviepy import VideoFileClip

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")
      
def fetch_image(ele: Dict[str, Union[str, Image.Image]]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    return image


def download_video(video_url: str) -> str:
    """下载视频到临时文件"""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # 创建临时文件
        file_ext = os.path.splitext(video_url)[1] or '.mp4'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Video download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Video download failed: {str(e)}")


def read_video_moviepy(video_path: str, target_fps: int = 25) -> Tuple[List[np.ndarray], List[float], float]:
    """
    使用MoviePy库读取视频帧和时间戳（增强版）
    
    Args:
        video_path: 视频文件路径
        target_fps: 目标帧率
    
    Returns:
        frames: 帧列表
        timestamps: 时间戳列表（单位：秒）
        actual_fps: 实际帧率
    """
    try:
        clip = VideoFileClip(video_path)
        
        # 获取视频原始信息
        original_fps = clip.fps
        duration = clip.duration
        total_frames = int(duration * original_fps)
        
        # 调整帧率
        if target_fps < original_fps:
            clip = clip.with_fps(target_fps)
            actual_fps = target_fps
        else:
            actual_fps = original_fps
        
        # 提取所有帧和时间戳
        frames = []
        timestamps = []
        
        for i, frame in enumerate(clip.iter_frames()):
            frames.append(frame)
            # 计算精确时间戳
            timestamp = i / actual_fps
            timestamps.append(timestamp)
        
        clip.close()
        
        print(f"MoviePy提取: {len(frames)} 帧, {actual_fps}fps, 总时长: {timestamps[-1]:.2f}秒")
        return frames, timestamps, actual_fps, clip.duration
        
    except Exception as e:
        raise ValueError(f"MoviePy读取失败: {e}")
    

# custom, no patching
def fetch_video(message):
    video = message.get("video") or message.get("video_url")
    fps = message.get("fps", 25)
    if video.startswith("http://") or video.startswith("https://"):
        video_path = download_video(video)
    elif video.startswith("file://"):
        video_path = video[7:]
    else:
        video_path = video

    frames, timestamps, actual_fps, duration = read_video_moviepy(video_path, target_fps=fps)
    if actual_fps != fps:
        logger.warning(f"Requested fps {fps} differs from actual fps {actual_fps}. Using actual fps.")
    return frames, timestamps, actual_fps, duration


def read_video_moviepy_generator(video_path: str, target_fps: int = 25):
    """
    使用MoviePy库以生成器方式读取视频帧和时间戳
    
    Args:
        video_path: 视频文件路径
        target_fps: 目标帧率
    
    Yields:
        frame: 视频帧 (numpy数组)
        timestamp: 时间戳 (单位：秒)
    
    Returns:
        actual_fps: 实际帧率
        duration: 视频总时长
    """
    try:
        clip = VideoFileClip(video_path)
        
        # 获取视频原始信息
        original_fps = clip.fps
        duration = clip.duration
        
        # 调整帧率
        if target_fps < original_fps:
            clip = clip.with_fps(target_fps)
            actual_fps = target_fps
        else:
            actual_fps = original_fps
        
        logger.debug(f"视频信息: {original_fps}fps -> {actual_fps}fps, 时长: {duration:.2f}秒")
        
        yield actual_fps, duration

        # 使用生成器逐帧返回
        frame_count = 0
        for frame in clip.iter_frames():
            timestamp = frame_count / actual_fps
            yield frame, timestamp
            frame_count += 1
        
        clip.close()
        logger.debug(f"MoviePy提取完成: {frame_count} 帧, {actual_fps}fps, 总时长: {timestamp:.2f}秒")

    except Exception as e:
        raise ValueError(f"MoviePy读取失败: {e}")
    
# custom, no patching
def fetch_video_generator(message):
    video = message.get("video") or message.get("video_url")
    fps = message.get("fps", 25)
    if video.startswith("http://") or video.startswith("https://"):
        video_path = download_video(video)
    elif video.startswith("file://"):
        video_path = video[7:]
    else:
        video_path = video

    frame_generator = read_video_moviepy_generator(video_path, target_fps=fps)
    actual_fps, duration = next(frame_generator)
    return frame_generator, actual_fps, duration
