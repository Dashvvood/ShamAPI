#!/usr/bin/env python3
"""
启动脚本：通过服务名启动对应的API服务
Usage: python start.py qwen3_vl [--workers 1] [--cuda-device 0]
"""
import os
import sys
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"


def get_available_services():
    """获取所有可用的服务"""
    if not APP_DIR.exists():
        return []
    
    services = []
    for item in APP_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            server_file = item / "server.py"
            if server_file.exists():
                services.append(item.name)
    return sorted(services)


def validate_service(service_name):
    """验证服务是否存在"""
    server_path = APP_DIR / service_name / "server.py"
    if not server_path.exists():
        print(f"Error: Service '{service_name}' not found")
        print(f"\nAvailable services:")
        for svc in get_available_services():
            print(f"  - {svc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start ShamAPI service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py qwen3_vl
  python start.py whisper --workers 2
  python start.py qwen3_vl --cuda-device 1 --workers 4
        """
    )
    
    parser.add_argument(
        'service',
        nargs='?',
        default='qwen3_vl',
        help='Service name to start (default: qwen3_vl)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help='Number of workers (default: 1)'
    )
    parser.add_argument(
        '-d', '--cuda-device',
        type=int,
        default=0,
        help='CUDA device ID (default: 0)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available services'
    )
    
    args = parser.parse_args()
    
    if args.list:
        services = get_available_services()
        print("Available services:")
        for svc in services:
            print(f"  - {svc}")
        return
    
    service_name = args.service
    validate_service(service_name)
    
    # 构建模块路径
    module_path = f"app.{service_name}.server:app"
    
    print(f"Starting service: {service_name}")
    print(f"Module path: {module_path}")
    print(f"Workers: {args.workers}")
    print(f"CUDA device: {args.cuda_device}")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    
    # 导入并启动 gunicorn
    try:
        import gunicorn.app.wsgiapp as wsgi
        sys.argv = [
            'gunicorn',
            '-c', str(PROJECT_ROOT / 'config' / 'gunicorn.conf.py'),
            '-w', str(args.workers),
            module_path
        ]
        wsgi.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == '__main__':
    main()

