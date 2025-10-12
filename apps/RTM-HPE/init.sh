#/bin/bash
cat >> .env << EOF
PROJECT_ROOT=$(dirname "$(dirname "$(pwd)")")
APP_ROOT=$(pwd)
TRANSFORMERS_OFFLINE=0
EOF

ln -s ../../data .