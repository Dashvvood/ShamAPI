import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

config_dir =os.environ.get("CONFIG", "development")
print(f"{config_dir = }")

WHISPER_CONFIG_PATH = os.path.join(PROJECT_ROOT, 
    f"config/{config_dir}/whisper.yaml")
WHISPERX_CONFIG_PATH = os.path.join(PROJECT_ROOT, 
    f"config/{config_dir}/whisperx.yaml")
RTMLIB_CONFIG_PATH = os.path.join(PROJECT_ROOT, 
    f"config/{config_dir}/rtmlib.yaml")
QWEN25_OMNI_CONFIG_PATH = os.path.join(PROJECT_ROOT, 
    f"config/{config_dir}/qwen25_omni.yaml")
QWEN3_VL_CONFIG_PATH = os.path.join(PROJECT_ROOT, 
    f"config/{config_dir}/qwen3_vl.yaml")
