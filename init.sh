#/bin/bash
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cat >> ${PROJECT_ROOT}/.env << EOF
PROJECT_ROOT=${PROJECT_ROOT}
EOF


function link_dirs() {
    local app_root=$1
    shift  # 移除第一个参数
    
    for dir in "$@"; do
        source=${PROJECT_ROOT}/${dir}
        target=${app_root}/${dir}
        if [ -L "${target}" ] || [ -e "${target}" ]; then
            echo "Skipping ${source} → ${target}"
        else
            echo "Create ${source} → ${target}"
            ln -s "${source}" "${target}"
        fi
    done
}

function dump_env() {
    local app_root=$1
    local app_name=$2
    
    cat > "${app_root}"/.env << EOF
PROJECT_ROOT=${PROJECT_ROOT}
APP_NAME=${app_name}
APP_ROOT=${app_root}
EOF
}

N=0
while IFS= read -r APP_ROOT; do
    APP_NAME=$(basename "${APP_ROOT}")
    dump_env "${APP_ROOT}" "${APP_NAME}"
    link_dirs "${APP_ROOT}" "cache" "data"
    ((N++))
done < <(find ${PROJECT_ROOT}/apps -maxdepth 1 -mindepth 1 -type d)

echo "Initialized ${N} apps under ${PROJECT_ROOT}/apps"
