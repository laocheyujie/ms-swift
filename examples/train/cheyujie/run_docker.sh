docker run -itd \
    --gpus all \
    --shm-size=64g \
    --net=host \
    -w /mnt/workspace \
    -v /data1/cheyujie/code/ms-swift:/mnt/workspace/ms-swift \
    -v /data1/cheyujie/datasets:/datasets \
    -v /data1/cheyujie/models:/models \
    -v /data1/cheyujie/output:/output \
    --name model \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3
