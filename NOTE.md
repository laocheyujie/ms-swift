# 整体流程

1. 权重转 MGT: 
2. 编辑训练脚本: `vi train.sh`
3. 启动训练: `nohup ./train.sh > train.log 2>&1 &`
4. 查看 Tensorboard
5. 把权重合并到主节点:
    1. 先进入 `--save` 所在路径
    2. 复制权重 `scp -r iter_0000100 node-1:/models/megatron_output/xxx/v1-20251023-200954`
6. 权重合并及转换 HF: [脚本](#lora)
7. 部署: sglang / vllm


# 安装

## 1. 下载 Megatron-LM
```bash
git clone git@github.com:NVIDIA/Megatron-LM.git Megatron-LM --branch core_r0.14.0
```
> 去 `swift/swift/megatron/init.py` 的 `init_megatron_env` 里查看当前具体使用的 mcore 版本

## 2. 下载 ms-swift
```bash
git clone git@github.com:laocheyujie/ms-swift.git
```

## 3. 下载镜像
```bash
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3
```

## 4. 启动容器
> [环境准备](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html#id1)


```bash
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
    --name ms \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3

docker exec -it ms /bin/bash
```

## 5. 容器内

### 基础依赖 (Multi-node)
```bash
apt update -y && apt install -y pdsh
apt install -y openssh-server

# 修改配置
sed -i 's/^#\?Port .*/Port 2333/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

export passwd=cheyujie && printf "${passwd}\n${passwd}\n"  | passwd root 

service ssh start

# 测试配置
/usr/sbin/sshd -t 
# 没有输出就是正常

ss -lntp | grep :2333

# ssh -p 2333 root@ip地址

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
```

### zsh
```bash
apt install -y zsh
wget https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh
chmod +x install.sh
./install.sh

# 补全提示
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# git clone git@github.com:zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# 高亮
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# git clone git@github.com:zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

vi +73 ~/.zshrc
plugins=(git zsh-autosuggestions zsh-syntax-highlighting pip)

source ~/.zshrc
```

### Python 依赖
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip

cd /mnt/workspace/ms-swift
git config --global --add safe.directory /mnt/workspace/ms-swift
pip install -e .
pip install swanlab ipykernel -U
```


### ~~hostfile~~
```bash
cd /mnt/workspace

tee hostfile <<-'EOF'
node-1 slots=8
node-2 slots=8
EOF
```


### ~~添加代理~~
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



### 检查网络
```bash
ibv_devinfo
```
```
hca_id:	mlx5_0
	transport:      InfiniBand (0)
	...
	link_layer:		Ethernet
```

Link layer:
- InfiniBand
- Ethernet: 用 Ethernet 以太网互连，表示该网卡当前被配置为以太网模式，而不是 InfiniBand 模式

Mellanox 的网卡（如 ConnectX-5、ConnectX-6 等）支持多种协议，包括：
- InfiniBand (IB)：一种高速网络协议，通常用于高性能计算（HPC）和数据中心。
- Ethernet：标准的以太网协议，广泛用于通用网络环境。RoCE（RDMA over Converged Ethernet），虽然可以显著降低延迟，但仍高于 IB。

如果某些机器是 "No IB devices found"，另一些机器又是 IB/RoCE，那就只能在使用 NCCL 时，禁用 IB/RoCE，否则无法正常跨机通信。
对应的NCCL环境变量为：
```bash
# 关闭IB
export NCCL_IB_DISABLE=1
# 关闭IBEXT(RoCE) 参考：https://github.com/NVIDIA/nccl/issues/676
# 注意必须要这个 不然关闭不彻底
export NCCL_IBEXT_DISABLE=1
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
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
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
2. packing: false (数据量呈现长文本较多时用 packing 可以，但是 packing 会稀释短文本的 gradient，可以先 packing 后取消)
3. 先确定训练数据的最大 Token Length，再调整 max_length


### LoRA 继续训练
1. 把最后一次 iter 的权重同步到所有节点上
2. 把主节点 LoRA 保存位置的 `args.json` 同步到所有节点上
3. 把 `latest_checkpointed_iteration.txt` 同步到所有节点上
4. 训练脚本增加 `--adapter_load /models/xxx/vxxx` 注意写到 iter 的上一层，脚本会读取该路径下的 `args.json` 和最后迭代数来加载 LoRA
5. 可以修改 `--dataset`, `--max_epochs` 等参数，但建议不要修改 LoRA 相关参数
6. `--finetune true` 会重开一个训练，迭代数重置为 0，不会跳过任何数据集；`--finetune false` 还没测试，应该是会加载之前的状态，继续训练


### 训练数据
```json
{
    "messages": [
        {"role": "system", "content": "<system>"}, 
        {"role": "user", "content": "<query1>"}, 
        {"role": "assistant", "content": "<response1>"}, 
        {"role": "user", "content": "<query2>"}, 
        {"role": "assistant", "content": "<response2>"}
    ]
}
```


### Tools / Agent 训练
#### 训练数据格式
```json
{
    "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", 
    "messages": [
        {"role": "user", "content": "北京和上海今天的天气情况"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, 
        {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, 
        {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, 
        {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}
    ]
}
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

#### 测试 Agent Template
```py
from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('ZhipuAI/GLM-4-9B-0414', load_model=False)
template = get_template(
    tokenizer.model_meta.template, 
    tokenizer, 
    # agent_template='hermes'
)
data = {
    "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", 
    "messages": [
        {"role": "user", "content": "北京和上海今天的天气情况"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, 
        {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, 
        {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, 
        {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}
    ]
}
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


## LoRA
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --mcore_adapters megatron_output/Qwen3-235B-A22B-Instruct-2507/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen3-235B-A22B-Instruct-2507/vx-xxx-hf
```

```bash
CUDA_VISIBLE_DEVICES=0 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters /models/megatron_output/GLM-4.5-Air-FH/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/megatron_output/GLM-4.5-Air-HF/vx \
    --test_convert_precision true
```
> - model_type: glm4_5


## 合并 LoRA
仅 Merge LoRA，而不转成 HF 格式权重:
```bash
CUDA_VISIBLE_DEVICES=0 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-mcore \
    --test_convert_precision true
```


## 权重转换 Megatron 转 HF

先把分散在不同机器上的权重移到一起

<a id="lora"></a>

### LoRA
```bash
CUDA_VISIBLE_DEVICES=0 \
MEGATRON_LM_PATH='/mnt/workspace/Megatron-LM' \
swift export \
    --model /models/ZhipuAI/GLM-4.5-Air \
    --mcore_model /models/ZhipuAI/GLM-4.5-Air-mcore \
    --mcore_adapters /models/megatron_output/GLM-4.5-Air-FH/vxxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /models/megatron_output/GLM-4.5-Air-HF/v59
```


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


## 部署
### NCCL 配置
```bash
which nvcc
nvcc --version
# 如果有信息，则不用进行后续安装

# 如果没有信息，需要安装 CUDA Toolkit
conda install -y -c nvidia cuda-toolkit=12.4

# 环境变量设置（当前 shell 生效）
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# 选择正确的 targets 目录（x86_64 / sbsa / aarch64 三选一自动判定）
if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  TARGET_DIR="$CUDA_HOME/targets/x86_64-linux"
elif [ -d "$CUDA_HOME/targets/sbsa-linux/include" ]; then
  TARGET_DIR="$CUDA_HOME/targets/sbsa-linux"
elif [ -d "$CUDA_HOME/targets/aarch64-linux/include" ]; then
  TARGET_DIR="$CUDA_HOME/targets/aarch64-linux"
else
  echo "ERROR: 未找到 CUDA targets 目录，请手动 ls $CUDA_HOME/targets"; exit 1
fi

# 让编译器能找到 CUDA 头文件/库
export CPATH="$TARGET_DIR/include:$CPATH"
export LIBRARY_PATH="$TARGET_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$TARGET_DIR/lib:$LD_LIBRARY_PATH"

# 一些构建系统会用到 $CONDA_PREFIX/include，这里做一个 include 软链映射（无 sudo 也可）
mkdir -p "$CUDA_HOME/include"
if [ ! -e "$CUDA_HOME/include/cuda_runtime.h" ]; then
  ln -snf "$TARGET_DIR/include" "$CUDA_HOME/include"
fi

# （可选）若有 sudo，很多三方会硬编码 /usr/local/cuda，做个软链最省心
# 没有 sudo 就跳过这步，问题也不大
if command -v sudo >/dev/null 2>&1; then
  sudo ln -snf "$CUDA_HOME" /usr/local/cuda
fi

# 验证
which nvcc
nvcc --version
```


### vLLM
#### 多机部署
多机部署需要在每台机器上设定 VLLM_HOST_IP，填入每个节点自己的 IP
```bash
export VLLM_HOST_IP=x.x.x.x
```

检测网络
```bash
NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=head_node_ip:8887 scripts/utils/check_nccl.py
```
> 每个节点都运行该命令
> [vLLM 多节点网络检测](https://docs.vllm.ai/en/latest/usage/troubleshooting.html#incorrect-hardwaredriver)

启动模型
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/megatron_output/GLM-4.5-Air-HF/vx \
    --served-model-name GLM-4.5-Air \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enforce-eager \
    --gpu_memory_utilization=0.9 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser glm4_moe \
    --reasoning-parser glm4_moe \
    --port 6060
```


### SGLang
#### 单机部署
```bash
python3 -m sglang.launch_server \
  --model /models/megatron_output/GLM-4.5-Air-HF/vx \
  --served-model-name GLM-4.5-Air \
  --context-length 131072 \
  --trust-remote-code \
  --tp 2 \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

如果报 `OSError: CUDA_HOME environment variable is not set` 错误，请从下面选择一种解决方式：

- `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>`

- `pip install flashinfer-python`


#### 多机部署
```bash
pip install sglang-router

# python -m sglang_router.launch_server --help
# python -m sglang_router.launch_router --help

# replace 172.16.4.52:20000 with your own node ip address and port of the first node

# Node 1
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 8 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 0

# Node 2
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 8 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 1
```

## 测试
```bash
curl http://localhost:6060/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "GLM-4.5-Air",
  "messages": [
    {"role": "user", "content": "你好，你是谁？"}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

# 新增模型
1. swift/llm/model/constant.py 新增 model 类型 `LLMModelType` | `BertModelType` | `RMModelType` | `MLLMModelType`
2. swift/llm/template/constant.py 新增 template 类型 `LLMTemplateType` | `RMTemplateType` | `MLLMTemplateType`
3. swift/llm/template/template 下面通过 `register_template(TemplateMeta)` 注册新增加的 template, `TemplateMeta`:
    1. `template_type`
    2. `template_cls`
    3. `prefix`, `suffix`, `prompt`, `default_system`, ...
4. swift/llm/model/model 下面通过 `register_model(ModelMeta)` 注册新增加的 model, `ModelMeta`:
    1. `model_type = LLMModelType.new_model_name`
    2. `model_groups`
    3. `template = TemplateType.new_model_name`
    4. `get_function`
    5. `architectures`
    6. `requires`: transformers 版本要求



# 报告
## SwanLab
安装：`pip install swanlab -i https://mirrors.aliyun.com/pypi/simple/`

sh 里添加：
- swanlab_token: SwanLab的api-key。
- swanlab_project: swanlab的project，需要在页面中预先创建好:[https://swanlab.cn/space/~](https://swanlab.cn/space/~)。
- swanlab_workspace: 默认为None，会使用api-key对应的username。
- swanlab_exp_name: 实验名，可以为空，为空时默认传入--output_dir的值。
- swanlab_lark_webhook_url: 默认为None。swanlab的lark webhook url，用于推送实验结果到飞书。
- swanlab_lark_secret: 默认为None。swanlab的lark secret，用于推送实验结果到飞书。
- swanlab_mode: 可选cloud和local，云模式或者本地模式。