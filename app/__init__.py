import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

config_dir =os.environ.get("CONFIG", "dev").lower()
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


def __getattr__(name: str):
    """懒加载：只有被访问时才导入对应模块，避免加载不需要的重型依赖。"""
    match name:
        case "qwen3_vl":
            from .qwen3_vl.server import app
            return app
        case "rtmlib":
            from .rtmlib.server import app
            return app
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")