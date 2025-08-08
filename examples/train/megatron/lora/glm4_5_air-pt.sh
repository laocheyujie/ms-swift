# thinking -> non-thinking
# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron pt \
    --load /models/ZhipuAI/GLM-4.5-Air-mcore \
    --dataset /mnt/workspace/ms-swift/datasets/pt.jsonl \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.00 \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 16 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save /models/megatron_output/GLM-4.5-Air-PT \
    --eval_interval 200 \
    --save_interval 400 \
    --packing true \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash


    # --pipeline_model_parallel_size 2 \

    # --optimizer_cpu_offload true \
    # --use_precision_aware_optimizer true

    # --model /models/ZhipuAI/GLM-4.5-Air \
    # --use_hf true \
    
    # --logging_steps 2 \

    # --report_to swanlab \
    # --swanlab_token xxxxxx \
    # --swanlab_project GLM-4.5-Air \
    # --swanlab_exp_name GLM-4.5-Air-PT \
    
    # --local_repo_path /mnt/workspace/.cache/modelscope/hub/_github \