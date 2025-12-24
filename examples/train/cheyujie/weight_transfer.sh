# 模型权重转换

# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --save Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true


# torch_dist -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --load Qwen3-30B-A3B-Instruct-2507-mcore \
    --save Qwen3-30B-A3B-Instruct-2507-hf \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true

# torch_dist -> safetensors (老版本全参训练后的权重转换)
```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model /models/megatron_output/GLM-4.5-Air-SFT/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/megatron_output/GLM-4.5-Air-HF/vx-xxx-hf \
    --test_convert_precision true
```
> - test_convert_precisio: 测试HF和Megatron格式权重转换的精度误差，若出现内存不足，请将`--test_convert_precision true`删除


# LoRA 权重转换
# torch_dist -> safetensors
# 只对 LoRA 权重进行类型转换 `--merge_lora false`
# - 基础模型是 safetensors 使用 `--model safetensors-path`
# - 基础模型是 torch_dist 使用 `--load torch-dist-path`
# - Adapter 是 safetensors 使用 `--adapters safetensors-path`
# - Adapter 是 torch_dist 使用 `--adapter_load torch-dist-path`
# NOTE: 注意 `--adapter_load`, `--adapters` 不要到 iter_xxx 层级，而是到上一层，并在上一层创建 `latest_checkpointed_iteration.txt`
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_load megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --merge_lora false \
    --to_hf true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true


# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-mcore \
    --merge_lora false \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true