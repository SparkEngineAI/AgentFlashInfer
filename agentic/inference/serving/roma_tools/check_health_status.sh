#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

MASTER_NODE=$(python3 ${SCRIPT_DIR}/get_master_ip.py)

if [[ -z ${POD_IP} ]]; then
    status_code=`python -c "import requests; print(requests.get('http://${MASTER_NODE}:8000/health').status_code)"`
else
    status_code=`python -c "import requests; print(requests.get('http://${MASTER_NODE}:9000/health').status_code)"`
fi

if [[ $status_code -eq 200 ]]; then
    echo "Service is alive."
    exit 0
else
    echo "Service is not alive"
    exit 1
fi