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
              '/datasets/train/General-Pretrain.jsonl' \
    --load_from_cache_file false \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --packing true \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-5}' \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 20 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /output/Che \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_only_model true \
    --deepspeed zero2_offload \
    --report_to tensorboard swanlab \
    --swanlab_project xxx \
    --swanlab_exp_name yyy


    # --save_total_limit 2 \
    # --system 'You are a helpful assistant.' \