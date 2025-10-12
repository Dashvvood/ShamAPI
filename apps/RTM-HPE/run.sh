#!/bin/bash

# 改进的虚拟环境管理脚本

# 检查虚拟环境是否已激活且是目标环境
if [ -n "$VIRTUAL_ENV" ]; then
    CURRENT_VENV=$(basename "$VIRTUAL_ENV")
    if [ "$CURRENT_VENV" = ".venv" ]; then
        echo "Virtual environment '$CURRENT_VENV' already activated."
    else
        echo "Another virtual environment '$CURRENT_VENV' is active."
        echo "Please deactivate it first."
        exit 1
    fi
else
    # 检查虚拟环境目录是否存在
    if [ -f ".venv/bin/activate" ]; then
        echo "Activating virtual environment..."
        
        echo "Virtual environment activated."
    else
        echo "Error: Virtual environment '.venv' not found."
        echo "Please create it first with: python -m venv .venv"
        exit 1
    fi
fi

# 启动Gunicorn服务器
echo "Starting Gunicorn server..."
gunicorn -c gunicorn_conf.py main:app
