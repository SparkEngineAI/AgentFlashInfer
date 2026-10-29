#!/bin/bash

copy_file() {
    src=$1
    dst=$2

    if cp ${src} ${dst} 2>/dev/null; then
        echo "cp ${src} ${dst} success!"
    else 
        if sudo cp ${src} ${dst} 2>/dev/null; then
            echo "sudo cp ${src} ${dst} success!"
        else
            echo "sudo cp ${src} ${dst} failed!"
            return 1
        fi
    fi
}

copy_vl_file() {
    copy_file ${PROJECT_DIR}/pangu_infer/models/vllm_ascend/processor_openpangu_vl.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/imageprocessor_openpangu_vl.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/videoprocessor_openpangu_vl.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/configuration_openpangu_vl.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/modeling_openpangu_vl.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/modeling_openpangu_embedded.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/rotary_embedding.py ${LOAD_CKPT_DIR}
}

copy_omni_file() {
    copy_file ${PROJECT_DIR}/pangu_infer/models/vllm_ascend/processor_openpangu_omni.py ${LOAD_CKPT_DIR}
    copy_file ${PROJECT_DIR}/pangu_infer/models/huggingface/configuration_openpangu_omni.py ${LOAD_CKPT_DIR}
    copy_vl_file
}