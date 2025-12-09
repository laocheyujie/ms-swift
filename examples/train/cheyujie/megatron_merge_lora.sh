# safetensors -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --adapters '/output/megatron_output/xxx/v1-20251112-204439/checkpoint-2695' \
    --output_dir /output/merged/xxx/v1 \
    --merge_lora true


# torch_dist -> safetensors
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters /models/megatron_output/xx/v2-20251104-195610 \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/merged/xxx/v2


