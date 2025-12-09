CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model /models/Qwen/Qwen3-32B \
    --adapters /output/xxx/v0-20251111-175936/checkpoint-336 \
    --output_dir /models/Che \
    --merge_lora true