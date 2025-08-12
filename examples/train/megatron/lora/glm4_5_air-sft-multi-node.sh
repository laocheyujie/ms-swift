
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
GLOO_SOCKET_IFNAME=eth0 \
NCCL_SOCKET_IFNAME=eth0 \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MASTER_ADDR=xx.xx.xx.xx \
MASTER_PORT=29500 \
megatron sft \
    --load /models/ZhipuAI/GLM-4.5-Air-mcore \
    --dataset '/mnt/workspace/ms-swift/datasets/FengHe-sft-qa.jsonl' \
              '/mnt/workspace/ms-swift/datasets/FengHe-sft-tools.jsonl' \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.00 \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 6 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save /models/megatron_output/GLM-4.5-Air-FH \
    --eval_interval 200 \
    --save_interval 400 \
    --max_length 128000 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --loss_scale default \
    --attention_backend flash
