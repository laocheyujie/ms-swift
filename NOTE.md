# 安装
## Docker 安装
### 启动容器
> [环境准备](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html#id1)

离线的话可以提前下好 Megatron-LM
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_r0.13.0
```

```bash
docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4


docker run --gpus all --shm-size=128g --net=host -itd -v /data/cheyujie/models:/models -v /data/cheyujie/code/ms-swift:/mnt/workspace/ms-swift -v /data/cheyujie/code/Megatron-LM:/mnt/workspace/Megatron-LM --name ms modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4 /bin/bash

docker exec -it ms /bin/bash
```


### 添加代理
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

### Python 依赖
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip

cd /mnt/workspace/ms-swift
pip install -e .
pip install 'transformers>=4.54' -U
pip install swanlab ipykernel -U
```


### Multi-node
```bash
apt update -y && apt install -y pdsh
apt install -y openssh-server

# 修改配置
sed -i 's/^#\?Port .*/Port 2333/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

export passwd=cheyujie && printf "${passwd}\n${passwd}\n"  | passwd root 

# 测试配置
/usr/sbin/sshd -t 
# 没有输出就是正常

service ssh start

ss -lntp | grep :2333

ssh -p 2333 root@ip地址

rm -rf ~/.ssh/*
ssh-keygen

tee ~/.ssh/config <<-'EOF'
Host node-1
        User  root
        Hostname 172.16.16.4
        port 2333
        IdentityFile ~/.ssh/id_rsa
Host node-2
        User  root
        Hostname 172.16.16.3
        port 2333
        IdentityFile ~/.ssh/id_rsa        
EOF


ssh-copy-id node-1
ssh-copy-id node-2

cd /mnt/workspace

tee hostfile <<-'EOF'
node-1 slots=8
node-2 slots=8
EOF
```

### 设置 NCCL
```bash
ip a

export GLOO_SOCKET_IFNAME=eth0
# 或 ib0，ensX，enpXsY… 选实际对外通信的那块
export NCCL_SOCKET_IFNAME=eth0
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
[环境准备](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html#id1)

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

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git config --global --add safe.directory /xxx/Megatron-LM
git checkout core_r0.13.0

export MEGATRON_LM_PATH='/xxx/Megatron-LM'
```

> 默认位置：local_repo_path: /mnt/workspace/.cache/modelscope/hub/_github/Megatron-LM



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
> - test_convert_precisio: 测试HF和Megatron格式权重转换的精度误差，若出现内存不足，请将`--test_convert_precision true`删除
> - thread_count: --to_mcore true时的模型切片数。默认为None，根据模型大小自动设置，使得最大分片小于10GB。
> - model_type: glm4_5




## 训练
### 训练流程

[Best Practices for SFT Training](https://github.com/modelscope/ms-swift/pull/5033)

[Best Practices for GRPO Training](https://github.com/modelscope/ms-swift/issues/4030)

注意：

1. 设置 MEGATRON_LM_PATH: `export MEGATRON_LM_PATH='/xxx/Megatron-LM'`

2. Multi-node: 添加`export GLOO_SOCKET_IFNAME=eth0`和`export NCCL_SOCKET_IFNAME=eth0`

3. 查看宿主机共享内存`df -h /dev/shm`


经验：

1. full: lr 1e-5, min_lr 1e-6; lora: lr 1e-4, min_lr 1e-5


## Tools
### 训练数据格式
```json
{"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "北京和上海今天的天气情况"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}]}
```

**注意**：
1. tools 值是一个包含 tool 列表的 json 字符串；messages 值是对话列表
2. messages 中 role 为 'tool_call' 和 'tool_response/tool' 的 content 部分都需要是 json 字符串
3. 支持并行调用工具 `{"role": "tool_call", "content": "..."}, {"role": "tool_call", "content": "..."}, {"role": "tool_response", "content": "..."}, {"role": "tool_response", "content": "..."}`


```py
import json

datasets = []

tools = [
    {
        "type": "function",
        "function": {
            "name": "realtime_aqi",
            "description": "天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如：上海"}
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "北京和上海今天的天气情况"},
    {
        "role": "tool_call",
        "content": '{"name": "realtime_aqi", "arguments": {"city": "北京"}}',
    },
    {
        "role": "tool_call",
        "content": '{"name": "realtime_aqi", "arguments": {"city": "上海"}}',
    },
    {
        "role": "tool_response",
        "content": '{"city": "北京", "aqi": "10", "unit": "celsius"}',
    },
    {
        "role": "tool_response",
        "content": '{"city": "上海", "aqi": "72", "unit": "fahrenheit"}',
    },
    {
        "role": "assistant",
        "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。",
    },
]

data = {
    "tools": json.dumps(tools, ensure_ascii=False),
    "messages": messages,
}
datasets.append(data)

for d in datasets:
    with open("tools.jsonl", "a") as f:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
```

### 测试
```py
from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('ZhipuAI/GLM-4-9B-0414', load_model=False)
template = get_template(
    tokenizer.model_meta.template, 
    tokenizer, 
    # agent_template='hermes'
)
data = {"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "北京和上海今天的天气情况"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}]}
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
```

### loss_scale
[loss_scale](https://swift.readthedocs.io/zh-cn/latest/Instruction/Agent%E6%94%AF%E6%8C%81.html#loss-scale)

#### ReACT
```json
{
    "Action:": [2.0, 2.0],
    "Action Input:": [2.0, 2.0],
    "Thought:": [1.0, 1.0],
    "Final Answer:": [1.0, 1.0],
    "Observation:": [2.0, 0.0]
}
```
每个列表的第一个值表示“字段本身”的权重，第二个值表示该字段“结果”的权重



## 权重转换 Megatron 转 HF
### Full
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

## LoRA
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --mcore_adapters megatron_output/Qwen3-235B-A22B-Instruct-2507/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen3-235B-A22B-Instruct-2507/vx-xxx-hf
```

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
> - model_type: glm4_5



## 推理

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model megatron_output/GLM-4.5-Air-HF/vx-xxx-hf \
    --infer_backend vllm \
    --stream true \
    --temperature 0 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048
```

非推理模式：
```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --max_model_len 8192 \
    --response_prefix '<think>\n\n</think>\n\n'
```









# 参数
## HF 转 Megatron
ExportArguments(model='/models/ZhipuAI/GLM-4.5-Air', model_type='glm4_5', model_revision=None, task_type='causal_lm', torch_dtype=torch.bfloat16, attn_impl=None, new_special_tokens=[], num_labels=None, problem_type=None, rope_scaling=None, device_map=None, max_memory={}, max_model_len=None, local_repo_path=None, init_strategy=None, template='glm4_5', system=None, max_length=2048, truncation_strategy='delete', max_pixels=None, agent_template=None, norm_bbox=None, use_chat_template=True, padding_free=False, padding_side='right', loss_scale='default', sequence_parallel_size=1, response_prefix=None, template_backend='swift', dataset=[], val_dataset=[], split_dataset_ratio=0.0, data_seed=42, dataset_num_proc=1, load_from_cache_file=True, dataset_shuffle=True, val_dataset_shuffle=False, streaming=False, interleave_prob=None, stopping_strategy='first_exhausted', shuffle_buffer_size=1000, download_mode='reuse_dataset_if_exists', columns={}, strict=False, remove_unused_columns=True, model_name=None, model_author=None, custom_dataset_info=[], quant_method=None, quant_bits=None, hqq_axis=None, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=None, max_new_tokens=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stream=False, stop_words=[], logprobs=False, top_logprobs=None, ckpt_dir=None, lora_modules=[], tuner_backend='peft', train_type='lora', adapters=[], external_plugins=[], seed=42, model_kwargs={}, load_args=True, load_data_args=False, packing=False, lazy_tokenize=False, cached_dataset=[], custom_register_path=[], use_hf=False, hub_token=None, ddp_timeout=18000000, ddp_backend=None, ignore_args_error=False, use_swift_lora=False, merge_lora=False, safe_serialization=True, max_shard_size='5GB', output_dir='/models/ZhipuAI/GLM-4.5-Air-mcore', quant_n_samples=256, quant_batch_size=1, group_size=128, to_cached_dataset=False, to_ollama=False, to_mcore=True, to_hf=False, mcore_model=None, mcore_adapters=[], thread_count=None, test_convert_precision=True, push_to_hub=False, hub_model_id=None, hub_private_repo=False, commit_message='update files', to_peft_format=False, exist_ok=False)

## Megatron 转 HF
args: ExportArguments(model='/models/ZhipuAI/GLM-4.5-Air', model_type='glm4_5', model_revision=None, task_type='causal_lm', torch_dtype=torch.bfloat16, attn_impl=None, new_special_tokens=[], num_labels=None, problem_type=None, rope_scaling=None, device_map=None, max_memory={}, max_model_len=None, local_repo_path=None, init_strategy=None, template='glm4_5', system=None, max_length=2048, truncation_strategy='delete', max_pixels=None, agent_template=None, norm_bbox=None, use_chat_template=False, padding_free=False, padding_side='right', loss_scale='default', sequence_parallel_size=1, response_prefix=None, template_backend='swift', dataset=[], val_dataset=[], split_dataset_ratio=0.0, data_seed=42, dataset_num_proc=1, load_from_cache_file=True, dataset_shuffle=True, val_dataset_shuffle=False, streaming=False, interleave_prob=None, stopping_strategy='first_exhausted', shuffle_buffer_size=1000, download_mode='reuse_dataset_if_exists', columns={}, strict=False, remove_unused_columns=True, model_name=None, model_author=None, custom_dataset_info=[], quant_method=None, quant_bits=None, hqq_axis=None, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=None, max_new_tokens=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stream=False, stop_words=[], logprobs=False, top_logprobs=None, ckpt_dir='/models/megatron_output/FengHe-GLM-4.5-Air/v7-20250804-230716', lora_modules=[], tuner_backend='peft', train_type='lora', adapters=[], external_plugins=[], seed=42, model_kwargs={}, load_args=True, load_data_args=False, packing=False, lazy_tokenize=False, cached_dataset=[], custom_register_path=[], use_hf=False, hub_token=None, ddp_timeout=18000000, ddp_backend=None, ignore_args_error=False, use_swift_lora=False, merge_lora=False, safe_serialization=True, max_shard_size='5GB', output_dir='/models/megatron_output/FengHe-GLM-4.5-Air-PT', quant_n_samples=256, quant_batch_size=1, group_size=128, to_cached_dataset=False, to_ollama=False, to_mcore=False, to_hf=True, mcore_model='/models/ZhipuAI/GLM-4.5-Air-mcore', mcore_adapters=['/models/megatron_output/FengHe-GLM-4.5-Air/v7-20250804-230716'], thread_count=None, test_convert_precision=True, push_to_hub=False, hub_model_id=None, hub_private_repo=False, commit_message='update files', to_peft_format=False, exist_ok=False)

