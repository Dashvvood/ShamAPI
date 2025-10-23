## Source Code
```python
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
```
- audios: 采样率16k/s
    - feature_size, 梅尔频谱波段
- images: 
- videos: 
    - `FPS=2.0`是固定在代码中的, patch=14, 
    - `nframes = total_frames / video_fps * fps`
