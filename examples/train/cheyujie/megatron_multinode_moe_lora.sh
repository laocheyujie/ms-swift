docker run -itd \
    --gpus all \
    --shm-size=128g \
    --net=host \
    -w /mnt/workspace \
    -v /data/cheyujie/code/ms-swift:/mnt/workspace/ms-swift \
    -v /data/cheyujie/code/Megatron-LM:/mnt/workspace/Megatron-LM \
    -v /data/cheyujie/datasets:/datasets \
    -v /data/cheyujie/models:/models \
    -v /data2/cheyujie/models:/output \
    --name swift \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3




PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
GLOO_SOCKET_IFNAME=eth0 \
NCCL_SOCKET_IFNAME=eth0 \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MASTER_ADDR=172.16.16.4 \
MASTER_PORT=29500 \
megatron sft \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset '/datasets/train/QA.jsonl' \
              '/datasets/train/General.jsonl' \
              '/datasets/train/MCQ.jsonl' \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --load_from_cache_file false \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save /output/megatron_output/Che \
    --eval_interval 20 \
    --save_interval 200 \
    --max_length 8192 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash