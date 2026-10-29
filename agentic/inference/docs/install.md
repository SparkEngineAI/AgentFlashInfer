# Pangu Infer 安装运行指南

## 软件配套

torch==2.5.1
torch_npu==2.5.1
vllm==0.9.1
vllm_ascend==0.9.1

## 获取 Docker 镜像

请选择与目标硬件和驱动版本兼容的 vLLM/vLLM-Ascend 镜像，并从组织认可的镜像仓库或发布渠道获取。

## 运行镜像

```bash
IMAGE=vllm-ascend-v0.9.1-a3-self-made-atb:1.0.1
# 1. 这里是挂载了全部卡，单卡改写--privileged为--device=/dev/davinci7
# 2. 需要按需修改倒数几行的磁盘文件挂载
docker run -itd --name vllm-ascend-env \
    --net=host \
    -u root \
    --privileged \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v "$PWD":"$PWD" \ # 工作目录
    -v /data:/data \ # 数据目录
    --entrypoint /bin/bash \
    $IMAGE
```

用以下指令进入镜像（注意需要以root进入获得修改vLLM版本的权限）：

```bash
docker exec -itu root vllm-ascend-env bash
```

## 运行Pangu Infer

修改`agentic/inference/serving/colocate_prefill`文件夹下对应的sh文件，如`agentic/inference/serving/colocate_prefill/qwen/serving_qwen.sh`中的相关参数，执行：

```
bash agentic/inference/serving/colocate_prefill/qwen/serving_qwen.sh
```

即可启动服务。
