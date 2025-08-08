# 安装
## Docker 安装
```bash
docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4
```
```bash
docker run --gpus all --shm-size=128g --net=host -itd -v /data/cheyujie/models:/models -v /data/cheyujie/github_fork/ms-swift:/mnt/workspace/ms-swift --name ms modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4 /bin/bash

docker exec -it ms /bin/bash
```

容器内：

添加代理：
`vi ~/.bashrc`
```bash
BASE_PROXY_URL="http://xxx.xxx.xxx.xxx:7890"

function proxyon() {
    export http_proxy="$BASE_PROXY_URL"
    export https_proxy="$BASE_PROXY_URL"
    echo "proxy started!"
}

function unproxy() {
    unset http_proxy
    unset https_proxy
    echo "proxy stoped!"
}

function proxytest() {
    echo "Testing proxy..."
    curl -I --proxy "$BASE_PROXY_URL" https://www.google.com 2>/dev/null | head -n 1
    if [ $? -eq 0 ]; then
        echo "Proxy test successful!"
    else
        echo "Proxy test failed!"
    fi
}
```


```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip

cd /mnt/workspace/ms-swift
pip install -e .
pip install 'transformers>=4.54' -U
pip install swanlab -U
```



## 手动安装
### 基础依赖
```bash
conda create -n swift python=3.10 -y
conda activate swift

pip install torch==2.6

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

### 安装 Flash-Atten
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.8.1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 扩展依赖
```bash
pip install swanlab -U

pip install "sglang[all]" -U
pip install deepspeed -U
pip install liger_kernel nvitop pre-commit math_verify py-spy -U

# pip install "vllm>=0.5.1" "transformers<4.54" "trl<0.20" -U
# pip install "lmdeploy>=0.5,<0.9" -U --no-deps
# pip install autoawq -U --no-deps
# pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
# pip install git+https://github.com/modelscope/ms-swift.git
# pip install timm -U
# pip install "deepspeed<0.17" -U
# pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
```


### Megatron 依赖
```bash
# 推荐torch版本：2.6
pip install pybind11

# transformer_engine
# 若出现安装错误，可以参考该issue解决: https://github.com/modelscope/ms-swift/issues/3793
pip install --no-build-isolation transformer_engine[pytorch]
# 或使用以下方式安装
# pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]

# apex
git clone https://github.com/NVIDIA/apex
cd apex
# https://github.com/modelscope/ms-swift/issues/4176
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# 如果版本校验有问题，可以在 setup.py 里，把 if (bare_metal_version != torch_binary_version) 代码块全部注释掉

# megatron-core
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0

# 若使用多机训练，请额外设置`MODELSCOPE_CACHE`环境变量为共享存储路径
# 这将确保数据集缓存共享，而加速预处理速度
expert MODELSCOPE_CACHE='/xxx/shared'

# Megatron-LM
# 依赖库Megatron-LM中的训练模块将由swift进行git clone并安装。你也可以通过环境变量`MEGATRON_LM_PATH`指向已经下载好的repo路径（断网环境，[core_r0.13.0分支](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.13.0)）。
export MEGATRON_LM_PATH='/xxx/Megatron-LM'
```


# Megatron
## 权重转换 HF 转 Megatron
```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /models/ZhipuAI/GLM-4.5-Air-mcore \
    --test_convert_precision true
```
> test_convert_precisio: 测试HF和Megatron格式权重转换的精度误差，若出现内存不足，请将`--test_convert_precision true`删除
> thread_count: --to_mcore true时的模型切片数。默认为None，根据模型大小自动设置，使得最大分片小于10GB。
> model_type: glm4_5


默认位置：
local_repo_path: /mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM
也可自行下载
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```
再指定
```bash
--local_repo_path xxx
```


## 训练
查看宿主机共享内存
```bash
df -h /dev/shm
```
经验：
1. full: lr 1e-5, min_lr 1e-6; lora: lr 1e-4, min_lr 1e-5
2. 如果离线，可以:
    1. `git clone https://github.com/NVIDIA/Megatron-LM.git`
    2. `git checkout core_r0.13.0`
    3. `export MEGATRON_LM_PATH='/xxx/Megatron-LM'`



## 权重转换 Megatron 转 HF
### Full
```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model /models/megatron_output/FengHe-GLM-4.5-Air/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/megatron_output/FengHe-GLM-4.5-Air/vx-xxx-hf \
    --test_convert_precision true
```
> test_convert_precisio: 测试HF和Megatron格式权重转换的精度误差，若出现内存不足，请将`--test_convert_precision true`删除

## LoRA
```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```
> model_type: glm4_5


swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters /models/megatron_output/FengHe-GLM-4.5-Air/v10-20250805-140427 \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/megatron_output/FengHe-GLM-4.5-Air-SFT \
    --test_convert_precision true


## 推理
```bash
swift infer \
    --model megatron_output/FengHe-GLM-4.5-Air/vx-xxx-hf \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

swift infer \
    --model /models/megatron_output/FengHe-GLM-4.5-Air-SFT \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048


# 参数
## HF 转 Megatron
ExportArguments(model='/models/ZhipuAI/GLM-4.5-Air', model_type='glm4_5', model_revision=None, task_type='causal_lm', torch_dtype=torch.bfloat16, attn_impl=None, new_special_tokens=[], num_labels=None, problem_type=None, rope_scaling=None, device_map=None, max_memory={}, max_model_len=None, local_repo_path=None, init_strategy=None, template='glm4_5', system=None, max_length=2048, truncation_strategy='delete', max_pixels=None, agent_template=None, norm_bbox=None, use_chat_template=True, padding_free=False, padding_side='right', loss_scale='default', sequence_parallel_size=1, response_prefix=None, template_backend='swift', dataset=[], val_dataset=[], split_dataset_ratio=0.0, data_seed=42, dataset_num_proc=1, load_from_cache_file=True, dataset_shuffle=True, val_dataset_shuffle=False, streaming=False, interleave_prob=None, stopping_strategy='first_exhausted', shuffle_buffer_size=1000, download_mode='reuse_dataset_if_exists', columns={}, strict=False, remove_unused_columns=True, model_name=None, model_author=None, custom_dataset_info=[], quant_method=None, quant_bits=None, hqq_axis=None, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=None, max_new_tokens=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stream=False, stop_words=[], logprobs=False, top_logprobs=None, ckpt_dir=None, lora_modules=[], tuner_backend='peft', train_type='lora', adapters=[], external_plugins=[], seed=42, model_kwargs={}, load_args=True, load_data_args=False, packing=False, lazy_tokenize=False, cached_dataset=[], custom_register_path=[], use_hf=False, hub_token=None, ddp_timeout=18000000, ddp_backend=None, ignore_args_error=False, use_swift_lora=False, merge_lora=False, safe_serialization=True, max_shard_size='5GB', output_dir='/models/ZhipuAI/GLM-4.5-Air-mcore', quant_n_samples=256, quant_batch_size=1, group_size=128, to_cached_dataset=False, to_ollama=False, to_mcore=True, to_hf=False, mcore_model=None, mcore_adapters=[], thread_count=None, test_convert_precision=True, push_to_hub=False, hub_model_id=None, hub_private_repo=False, commit_message='update files', to_peft_format=False, exist_ok=False)

## Megatron 转 HF
args: ExportArguments(model='/models/ZhipuAI/GLM-4.5-Air', model_type='glm4_5', model_revision=None, task_type='causal_lm', torch_dtype=torch.bfloat16, attn_impl=None, new_special_tokens=[], num_labels=None, problem_type=None, rope_scaling=None, device_map=None, max_memory={}, max_model_len=None, local_repo_path=None, init_strategy=None, template='glm4_5', system=None, max_length=2048, truncation_strategy='delete', max_pixels=None, agent_template=None, norm_bbox=None, use_chat_template=False, padding_free=False, padding_side='right', loss_scale='default', sequence_parallel_size=1, response_prefix=None, template_backend='swift', dataset=[], val_dataset=[], split_dataset_ratio=0.0, data_seed=42, dataset_num_proc=1, load_from_cache_file=True, dataset_shuffle=True, val_dataset_shuffle=False, streaming=False, interleave_prob=None, stopping_strategy='first_exhausted', shuffle_buffer_size=1000, download_mode='reuse_dataset_if_exists', columns={}, strict=False, remove_unused_columns=True, model_name=None, model_author=None, custom_dataset_info=[], quant_method=None, quant_bits=None, hqq_axis=None, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=None, max_new_tokens=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stream=False, stop_words=[], logprobs=False, top_logprobs=None, ckpt_dir='/models/megatron_output/FengHe-GLM-4.5-Air/v7-20250804-230716', lora_modules=[], tuner_backend='peft', train_type='lora', adapters=[], external_plugins=[], seed=42, model_kwargs={}, load_args=True, load_data_args=False, packing=False, lazy_tokenize=False, cached_dataset=[], custom_register_path=[], use_hf=False, hub_token=None, ddp_timeout=18000000, ddp_backend=None, ignore_args_error=False, use_swift_lora=False, merge_lora=False, safe_serialization=True, max_shard_size='5GB', output_dir='/models/megatron_output/FengHe-GLM-4.5-Air-PT', quant_n_samples=256, quant_batch_size=1, group_size=128, to_cached_dataset=False, to_ollama=False, to_mcore=False, to_hf=True, mcore_model='/models/ZhipuAI/GLM-4.5-Air-mcore', mcore_adapters=['/models/megatron_output/FengHe-GLM-4.5-Air/v7-20250804-230716'], thread_count=None, test_convert_precision=True, push_to_hub=False, hub_model_id=None, hub_private_repo=False, commit_message='update files', to_peft_format=False, exist_ok=False)


-------------------- end of arguments ---------------------
INFO:megatron.core.num_microbatches_calculator:setting number of microbatches to constant 16
> setting tensorboard ...
WARNING: one_logger package is required to enable e2e metrics tracking. please go to https://confluence.nvidia.com/display/MLWFO/Package+Repositories for details to install it
WARNING:megatron.core.rerun_state_machine:RerunStateMachine initialized in mode disabled
torch distributed is already initialized, skipping initialization ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 42 ...
> compiling dataset index builder ...
make: 进入目录“/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM/megatron/core/datasets”
g++ -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -I/usr/local/include/python3.10 -I/usr/local/lib/python3.10/site-packages/pybind11/include helpers.cpp -o helpers_cpp.cpython-310-x86_64-linux-gnu.so
In file included from helpers.cpp:12:
/usr/local/lib/python3.10/site-packages/pybind11/include/pybind11/numpy.h: In function ‘void build_exhaustive_blending_indices(pybind11::array_t<short int>&, pybind11::array_t<long int>&, const pybind11::array_t<long int>&, int32_t)’:
/usr/local/lib/python3.10/site-packages/pybind11/include/pybind11/numpy.h:633:14: warning: ‘error_argmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
  633 |     return i * strides[Dim] + byte_offset_unsafe<Dim + 1>(strides, index...);
      |            ~~^~~~~~~~~~
helpers.cpp:49:13: note: ‘error_argmax’ was declared here
   49 |     int64_t error_argmax;
      |             ^~~~~~~~~~~~
make: 离开目录“/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM/megatron/core/datasets”
>>> done with dataset index builder. Compilation time: 7.948 seconds
WARNING: constraints for invoking optimized fused softmax kernel are not met. We default back to unfused kernel invocations.
> compiling and loading fused kernels ...
[rank0]:[W804 19:19:43.148778003 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
>>> done with compiling and loading fused kernels. Compilation time: 0.404 seconds
building GPT model ...
/mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM/megatron/core/transformer/transformer_config.py:837: UserWarning: If you are using transformer_engine as the transformer implementation, the core_attn is from transformer_engine and may be the fused version. For fused attention, you have no need to set 'core_attn' to recompute. Please check that the core_attn recompute is really needed.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/transformer_engine/pytorch/cpu_offload.py:670: DeprecationWarning: Offloading weights is deprecated. Using offload_weights=True does not have any effect.
  warnings.warn(