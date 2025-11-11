docker run -itd \
    --gpus all \
    --shm-size=64g \
    --net=host \
    -w /mnt/workspace \
    -v /data/cheyujie/code/ms-swift:/mnt/workspace/ms-swift \
    -v /data/cheyujie/datasets:/datasets \
    -v /data/cheyujie/models:/models \
    -v /data/cheyujie/output:/output \
    --name swift \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3




nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /models/Qwen/Qwen3-32B \
    --dataset '/datasets/train/Pseudo-Pretrain.jsonl' \
              '/datasets/train/General.jsonl' \
              '/datasets/train/Self-Cognition.jsonl' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --packing true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 20 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/Che/v0 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_only_model true \
    --deepspeed zero2


    # --save_total_limit 2 \
    # --system 'You are a helpful assistant.' \