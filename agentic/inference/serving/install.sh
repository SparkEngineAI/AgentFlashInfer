#!/usr/bin/env bash

# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 上移一层到父目录
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

echo "Project dir: ${PROJECT_DIR}"

cd $PROJECT_DIR

pip install -r $PROJECT_DIR/requirements.txt