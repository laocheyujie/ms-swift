# safetensors -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --adapters '/output/megatron_output/xxx/vx-xxx/checkpoint-2695' \
    --output_dir /output/merged/xxx/v1 \
    --merge_lora true


# torch_dist -> safetensors
# 先把分散在不同机器上的权重移到一起
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters /models/megatron_output/xx/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/merged/xxx/v2 \
    --test_convert_precision true
# - model_type: glm4_5


# torch_dist -> torch_dist
# 仅 Merge LoRA，而不转成 HF 格式权重
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --mcore_adapters /models/megatron_output/xxx/vx-xxx \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-mcore \
    --merge_lora true \
    --test_convert_precision true