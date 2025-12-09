import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

WHISPER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/whisper.yaml")
WHISPERX_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/whisperx.yaml")
RTMLIB_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/rtmlib.yaml")
QWEN25_OMNI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/qwen25_omni.yaml")
QWEN3_VL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/qwen3_vl.yaml")