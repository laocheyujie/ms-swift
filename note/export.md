## convert_hf2mcore

### 流程
1. `hf_model, template = prepare_model_template(args)`
    1. `model, processor = args.get_model_processor(**kwargs)`
        1. `BaseArguments.get_model_tokenizer(**kwargs)`
            1. `model_info, model_meta = get_model_info_meta(...)`
                1. `model_meta = get_matched_model_meta(model_id_or_path)`
                2. `model_info = _get_model_info(...)`
                3. `model_meta = MODEL_MAPPING[model_type]`
                4. `return model_info, model_meta`
            2. `model, processor = get_function(...)` 实际调用的是 `get_model_tokenizer_with_flash_attn`
                1. `model_config = AutoConfig.from_pretrained(...)`
                2. `model, processor = get_model_tokenizer_from_local(...)`
                    1. `tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)`
                    2. `model = AutoModelForCausalLM.from_pretrained(...)`
                    3. tokenizer pad_token, eos_token 替换
                    4. `return model, tokenizer`
                3. `reutrn model, tokenizer`
            3. `tokenizer = processor.tokenizer or processor`
            4. add_special_tokens
            5. `return model, processor`
    2. `template = BaseArguments.get_template(processor)` template 里包含了 processor
        1. `template_kwargs = BaseArguments.get_template_kwargs()`
        2. `template = get_template(template_type, processor, **template_kwargs)`
            1. `template_meta = TEMPLATE_MAPPING[template_type]`
            2. 实例化 `template = template_meta.template_cls(...)`, e.g. `<class 'swift.llm.template.template.utils.ThinkingTemplate'>`
            3. `return template`
        3. `return template`
    3. `return model, template`
2. `processor = template.processor`
3. 计算 Megatron 格式需要的 shard 数
4. `megatron_model_meta = get_megatron_model_meta(args.model_type)`
5. HF config 转 Megatron config `kwargs = megatron_model_meta.convert_hf_config(processor.model_info.config)` 实际使用`swift/megatron/model/gpt/config.py convert_gpt_hf_config`:
    1. `convert_hf_config` 把 HF `config` 映射为 `megatron_config`
    2. 获取模型的架构 `architectures`
    3. 不同的模型加不同的属性补丁
6. 根据 `megatron_config` 实例化 `MegatronArguments`
7. 将 megatron 的 tokenizer 替换成 hf 本身的 tokenzier
8. `extra_args = megatron_args.parse_to_megatron()`
    1. 把 `megatron_args` 转换成命令行格式的参数并加入 `sys.argv`
    2. 返回 `extra_args`
9. 初始化 Megatron 环境 `initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)`
10. 获取 Megatron Model `mg_model = megatron_model_meta.model_provider()`
    1. `args = get_args()` 使用 Megatron-LM 自带的函数读取所有配置的
    2. `config = core_transformer_config_from_args(args)` 使用 Megatron-LM 自带的配置读取功能得到 `config`，把激活函数、初始化方法参数绑定到具体的函数
    3. `transformer_layer_spec = get_gpt_decoder_block_spec(...)` 得到 Decoder Block Spec
        1. `dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)` 
            1. `backend = TESpecProvider()`
            2. `mlp = get_mlp_module_spec_for_backend(...)`
                1. return `ModuleSpec(module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2))`
        2. `moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)`
            1. `backend = TESpecProvider()`
            2. `mlp = get_mlp_module_spec_for_backend(...)`
                1. `get_moe_module_spec_for_backend(...)` 返回 MoE Module Spec
        3. 根据 `config.num_layers` 和 `moe_layer_pattern` 来构建 GPT 的所有 Transformer 层并返回
    4. 返回 `GPTModel` 实例
11. `megatron_model_meta.convert_hf2mcore(hf_model, mg_model)` 进行对应权重的拷贝
12. `test_convert_precision(hf_model, mg_model, template)` 对比两类模型的精度差别
    1. 计算参数个数，参数大小
    2. 把测试的对话 encode 后分别跑两类模型计算 logits
    3. 根据 logits 选出最大可能性的 tokens 进行比较
13. 删除 hf_model
14. `args.save_args()` 保存参数
15. `mg_save_checkpoint(1, [mg_model], None, None, 0)` 保存 Megatron 模型


### model_info
ModelInfo(
    model_type='glm4_5', 
    model_dir='/models/ZhipuAI/GLM-4.5-Air', 
    torch_dtype=torch.bfloat16, 
    max_model_len=131072, 
    quant_method=None, 
    quant_bits=None, 
    rope_scaling=None, 
    is_moe_model=True, 
    config=None, 
    task_type=None, 
    num_labels=None
)

### model_meta
ModelMeta(
    model_type='glm4_5', 
    model_groups=[
        ModelGroup(
            models=[
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5-Air-Base', 
                    hf_model_id='zai-org/GLM-4.5-Air-Base', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                ), 
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5-Air', 
                    hf_model_id='zai-org/GLM-4.5-Air', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                ), 
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5-Air-FP8', 
                    hf_model_id='zai-org/GLM-4.5-Air-FP8', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                ), 
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5-Base', 
                    hf_model_id='zai-org/GLM-4.5-Base', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                ), 
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5', 
                    hf_model_id='zai-org/GLM-4.5', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                ), 
                Model(
                    ms_model_id='ZhipuAI/GLM-4.5-FP8', 
                    hf_model_id='zai-org/GLM-4.5-FP8', 
                    model_path=None, 
                    ms_revision=None, 
                    hf_revision=None
                )
            ], 
            ignore_patterns=None, 
            requires=None, 
            tags=[]
        )
    ], 
    template='glm4_5', 
    get_function=<function get_model_tokenizer_with_flash_attn at 0x7fccd3c47ec0>, 
    model_arch=None, 
    architectures=['Glm4MoeForCausalLM'], 
    additional_saved_files=[], 
    torch_dtype=None, 
    is_multimodal=False, 
    is_reward=False, 
    task_type=None, 
    ignore_patterns=None, 
    requires=['transformers>=4.54'], 
    tags=[]
)

### model_config
Glm4MoeConfig {
  "architectures": [
    "Glm4MoeForCausalLM"
  ],
  "attention_bias": true,
  "attention_dropout": 0.0,
  "eos_token_id": [
    151329,
    151336,
    151338
  ],
  "first_k_dense_replace": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 10944,
  "max_position_embeddings": 131072,
  "model_type": "glm4_moe",
  "moe_intermediate_size": 1408,
  "n_group": 1,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 96,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 46,
  "num_key_value_heads": 8,
  "num_nextn_predict_layers": 1,
  "pad_token_id": 151329,
  "partial_rotary_factor": 0.5,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "routed_scaling_factor": 1.0,
  "tie_word_embeddings": false,
  "topk_group": 1,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.54.1",
  "use_cache": true,
  "use_qk_norm": false,
  "vocab_size": 151552
}

### template_kwargs
{
    'default_system': None, 
    'max_length': 2048, 
    'truncation_strategy': 'raise', 
    'max_pixels': None, 
    'agent_template': None, 
    'norm_bbox': None, 
    'use_chat_template': True, 
    'remove_unused_columns': True, 
    'padding_free': False, 
    'padding_side': 'right', 
    'loss_scale': 'default', 
    'sequence_parallel_size': 1, 
    'response_prefix': None, 
    'template_backend': 'swift'
}

### template_meta
GLM4_5TemplateMeta(
    template_type='glm4_5', 
    prefix=['[gMASK]<sop>'], 
    prompt=['<|user|>\n{{QUERY}}<|assistant|>\n'], 
    chat_sep=[], 
    suffix=['<|user|>'], 
    template_cls=<class 'swift.llm.template.template.utils.ThinkingTemplate'>, 
    system_prefix=['[gMASK]<sop><|system|>\n{{SYSTEM}}'], 
    default_system=None, 
    response_prefix='', 
    auto_add_bos=True, 
    stop_words=[
        '<|endoftext|>', 
        '<|user|>', 
        '<|observation|>'
    ], 
    agent_template='glm4_5'
)


### MEGATRON_MODEL_MAPPING
{
    'gpt': MegatronModelMeta(
        megatron_model_type='gpt', 
        model_types=[
            'qwen2', 
            'qwen2_5', 
            'qwq', 
            'qwq_preview', 
            'qwen2_5_math', 
            'llama', 
            'llama3', 
            'llama3_1', 
            'llama3_2', 
            'longwriter_llama3_1', 
            'codefuse_codellama', 
            'marco_o1', 
            'deepseek', 
            'deepseek_r1_distill', 
            'yi', 
            'yi_coder', 
            'sus', 
            'skywork_o1', 
            'openbuddy_llama', 
            'openbuddy_llama3', 
            'megrez', 
            'reflection', 
            'numina', 
            'ziya', 
            'mengzi3', 
            'qwen3', 
            'qwen3_thinking', 
            'qwen2_moe', 
            'qwen3_moe', 
            'qwen3_moe_thinking', 
            'internlm3', 
            'mimo', 
            'mimo_rl', 
            'moonlight', 
            'deepseek_moe', 
            'deepseek_v2', 
            'deepseek_v2_5', 
            'deepseek_r1', 
            'dots1', 
            'ernie', 
            'glm4_5'
        ], 
        model_provider=<function model_provider at 0x7fcb740feac0>, 
        convert_hf_config=<function convert_gpt_hf_config at 0x7fcb740fdd00>, 
        convert_mcore2hf=<function convert_mcore2hf at 0x7fcb740fe980>, 
        convert_hf2mcore=<function convert_hf2mcore at 0x7fcb740fe520>, 
        extra_args_provider=None
    )
}

### _MODEL_META_MAPPING
{
    'qwen2': 'gpt', 
    'qwen2_5': 'gpt', 
    'qwq': 'gpt', 
    'qwq_preview': 'gpt', 
    'qwen2_5_math': 'gpt', 
    'llama': 'gpt', 
    'llama3': 'gpt', 
    'llama3_1': 'gpt', 
    'llama3_2': 'gpt', 
    'longwriter_llama3_1': 'gpt', 
    'codefuse_codellama': 'gpt', 
    'marco_o1': 'gpt', 
    'deepseek': 'gpt', 
    'deepseek_r1_distill': 'gpt', 
    'yi': 'gpt', 
    'yi_coder': 'gpt', 
    'sus': 'gpt', 
    'skywork_o1': 'gpt', 
    'openbuddy_llama': 'gpt', 
    'openbuddy_llama3': 'gpt', 
    'megrez': 'gpt', 
    'reflection': 'gpt', 
    'numina': 'gpt', 
    'ziya': 'gpt', 
    'mengzi3': 'gpt', 
    'qwen3': 'gpt', 
    'qwen3_thinking': 'gpt', 
    'qwen2_moe': 'gpt', 
    'qwen3_moe': 'gpt', 
    'qwen3_moe_thinking': 'gpt', 
    'internlm3': 'gpt', 
    'mimo': 'gpt', 
    'mimo_rl': 'gpt', 
    'moonlight': 'gpt', 
    'deepseek_moe': 'gpt', 
    'deepseek_v2': 'gpt', 
    'deepseek_v2_5': 'gpt', 
    'deepseek_r1': 'gpt', 
    'dots1': 'gpt', 
    'ernie': 'gpt', 
    'glm4_5': 'gpt'
}

### megatron_model_meta
MegatronModelMeta(
    megatron_model_type='gpt', 
    model_types=[
        'qwen2', 
        'qwen2_5', 
        'qwq', 
        'qwq_preview', 
        'qwen2_5_math', 
        'llama', 
        'llama3', 
        'llama3_1', 
        'llama3_2', 
        'longwriter_llama3_1', 
        'codefuse_codellama', 
        'marco_o1', 
        'deepseek', 
        'deepseek_r1_distill', 
        'yi', 
        'yi_coder', 
        'sus', 
        'skywork_o1', 
        'openbuddy_llama', 
        'openbuddy_llama3', 
        'megrez', 
        'reflection', 
        'numina', 
        'ziya', 
        'mengzi3', 
        'qwen3', 
        'qwen3_thinking', 
        'qwen2_moe', 
        'qwen3_moe', 
        'qwen3_moe_thinking', 
        'internlm3', 
        'mimo', 
        'mimo_rl', 
        'moonlight', 
        'deepseek_moe', 
        'deepseek_v2', 
        'deepseek_v2_5', 
        'deepseek_r1', 
        'dots1', 
        'ernie', 
        'glm4_5'
    ], 
    model_provider=<function model_provider at 0x7fcb740feac0>, 
    convert_hf_config=<function convert_gpt_hf_config at 0x7fcb740fdd00>, 
    convert_mcore2hf=<function convert_mcore2hf at 0x7fcb740fe980>, 
    convert_hf2mcore=<function convert_hf2mcore at 0x7fcb740fe520>, 
    extra_args_provider=None
)

### megatron_config
{
    'num_layers': 46, 
    'hidden_size': 4096, 
    'ffn_hidden_size': 10944, 
    'num_attention_heads': 96, 
    'num_query_groups': 8, 
    'max_position_embeddings': 131072, 
    'norm_epsilon': 1e-05, 
    'rotary_base': 1000000, 
    'padded_vocab_size': 151552, 
    'attention_dropout': 0.0, 
    'untie_embeddings_and_output_weights': True, 
    'swiglu': True, 
    'add_qkv_bias': True, 
    'kv_channels': 128, 
    'architectures': ['Glm4MoeForCausalLM'], 
    'moe_ffn_hidden_size': 1408, 
    'moe_router_topk': 8, 
    'num_experts': 128, 
    'moe_router_pre_softmax': False, 
    'moe_router_topk_scaling_factor': 1.0, 
    'qk_layernorm': False, 
    'partial_rotary_factor': 0.5
}

打了模型补丁后 megatron_config:
{
    'num_layers': 46, 
    'hidden_size': 4096, 
    'ffn_hidden_size': 10944, 
    'num_attention_heads': 96, 
    'num_query_groups': 8, 
    'max_position_embeddings': 131072, 
    'norm_epsilon': 1e-05, 
    'rotary_base': 1000000, 
    'padded_vocab_size': 151552, 
    'attention_dropout': 0.0, 
    'untie_embeddings_and_output_weights': True, 
    'swiglu': True, 
    'add_qkv_bias': True, 
    'kv_channels': 128, 
    'architectures': 'Glm4MoeForCausalLM', 
    'moe_ffn_hidden_size': 1408, 
    'moe_router_topk': 8, 
    'num_experts': 128, 
    'moe_router_pre_softmax': False, 
    'moe_router_topk_scaling_factor': 1.0, 
    'qk_layernorm': False, 
    'partial_rotary_factor': 0.5, 
    'moe_router_score_function': 'sigmoid', 
    'moe_layer_freq': '[0]*1+[1]*45', 
    'moe_router_enable_expert_bias': True, 
    'moe_shared_expert_intermediate_size': 1408
}

转换成命令行后的 new_args:
['--micro-batch-size', '1', '--global-batch-size', '16', '--recompute-granularity', 'selective', '--recompute-modules', 'core_attn', '--use-cpu-initialization', '--log-interval', '5', '--tensorboard-dir', '/models/ZhipuAI/GLM-4.5-Air-Debug/runs', '--cross-entropy-fusion-impl', 'native', '--calculate-per-token-loss', '--attention-backend', 'unfused', '--optimizer', 'adam', '--optimizer-offload-fraction', '1.0', '--main-grads-dtype', 'fp32', '--main-params-dtype', 'fp32', '--exp-avg-dtype', 'fp32', '--exp-avg-sq-dtype', 'fp32', '--dataloader-type', 'cyclic', '--manual-gc-interval', '0', '--lr', '1e-05', '--lr-decay-style', 'cosine', '--lr-warmup-iters', '0', '--min-lr', '0', '--weight-decay', '0.1', '--clip-grad', '1.0', '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--adam-eps', '1e-08', '--sgd-momentum', '0.9', '--save', '/models/ZhipuAI/GLM-4.5-Air-Debug', '--save-interval', '500', '--no-save-optim', '--no-save-rng', '--no-load-optim', '--no-load-rng', '--ckpt-format', 'torch_dist', '--no-initialization', '--auto-detect-ckpt-format', '--exit-on-missing-checkpoint', '--distributed-backend', 'nccl', '--local-rank', '0', '--use-distributed-optimizer', '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1', '--context-parallel-size', '1', '--distributed-timeout-minutes', '300000', '--num-layers', '46', '--hidden-size', '4096', '--ffn-hidden-size', '10944', '--num-attention-heads', '96', '--group-query-attention', '--num-query-groups', '8', '--max-position-embeddings', '131072', '--position-embedding-type', 'rope', '--rotary-base', '1000000', '--rotary-percent', '1.0', '--normalization', 'RMSNorm', '--norm-epsilon', '1e-05', '--swiglu', '--untie-embeddings-and-output-weights', '--disable-bias-linear', '--add-qkv-bias', '--attention-dropout', '0.0', '--hidden-dropout', '0.0', '--kv-channels', '128', '--transformer-impl', 'transformer_engine', '--num-experts', '128', '--moe-layer-freq', '[0]*1+[1]*45', '--moe-ffn-hidden-size', '1408', '--moe-shared-expert-intermediate-size', '1408', '--moe-router-topk', '8', '--moe-router-dtype', 'fp32', '--moe-router-score-function', 'sigmoid', '--moe-router-bias-update-rate', '0.001', '--moe-router-enable-expert-bias', '--moe-router-topk-scaling-factor', '1.0', '--moe-router-load-balancing-type', 'aux_loss', '--expert-model-parallel-size', '1', '--moe-token-dispatcher-type', 'alltoall', '--moe-grouped-gemm', '--moe-aux-loss-coeff', '0.0', '--moe-token-drop-policy', 'probs', '--kv-lora-rank', '32', '--qk-head-dim', '128', '--qk-pos-emb-head-dim', '64', '--fp8-recipe', 'delayed', '--fp8-amax-history-len', '1024', '--fp8-amax-compute-algo', 'max', '--bf16', '--attention-softmax-in-fp32', '--tensorboard-log-interval', '1', '--tensorboard-queue-size', '50', '--log-timers-to-tensorboard', '--log-validation-ppl-to-tensorboard', '--log-memory-to-tensorboard', '--eval-iters', '-1', '--eval-interval', '500', '--seed', '42', '--seq-length', '131072', '--num-workers', '4']

### extra_args
{
    'train_type': 'full', 
    'freeze_parameters': [], 
    'freeze_parameters_regex': None, 
    'freeze_parameters_ratio': 0.0, 
    'trainable_parameters': [], 
    'trainable_parameters_regex': None, 
    'adapter_load': None, 
    'target_modules': ['all-linear'], 
    'target_regex': None, 
    'modules_to_save': [], 
    'lora_rank': 8, 
    'lora_alpha': 32, 
    'lora_dropout': 0.05, 
    'lora_bias': 'none', 
    'lora_dtype': None, 
    'use_rslora': False, 
    'ref_load': None, 
    'beta': 0.1, 
    'rpo_alpha': None, 
    'reference_free': False, 
    'label_smoothing': 0.0, 
    'f_divergence_type': 'reverse_kl', 
    'loss_type': 'sigmoid', 
    'padded_vocab_size': 151552, 
    'initialize_embedding': False, 
    'rope_scaling': None, 
    'torch_dtype': torch.bfloat16, 
    'padding_free': True, 
    'mlp_padding_free': False, 
    'dataloader_persistent_workers': True, 
    'dataloader_prefetch_factor': 10, 
    'architectures': 'Glm4MoeForCausalLM', 
    'max_epochs': None, 
    'enable_dft_loss': False, 
    'original_max_position_embeddings': None, 
    'partial_rotary_factor': 0.5, 
    'use_shared_expert_gate': False
}


### initialize_megatron args
Namespace(num_layers=46, encoder_num_layers=None, decoder_num_layers=None, hidden_size=4096, ffn_hidden_size=10944, num_attention_heads=96, attention_backend=<AttnBackend.unfused: 3>, kv_channels=128, group_query_attention=True, num_query_groups=8, max_position_embeddings=131072, position_embedding_type='rope', relative_attention_num_buckets=32, relative_attention_max_distance=128, use_rotary_position_embeddings=False, rotary_base=1000000, rotary_percent=1.0, rotary_interleaved=False, rotary_seq_len_interpolation_factor=None, use_rope_scaling=False, rope_scaling_factor=8.0, no_rope_freq=None, add_position_embedding=True, mrope_section=None, make_vocab_size_divisible_by=128, normalization='RMSNorm', norm_epsilon=1e-05, apply_layernorm_1p=False, apply_residual_connection_post_layernorm=False, openai_gelu=False, squared_relu=False, swiglu=True, onnx_safe=None, bert_binary_head=True, untie_embeddings_and_output_weights=True, multi_latent_attention=False, mtp_num_layers=None, mtp_loss_scaling_factor=0.1, attention_dropout=0.0, hidden_dropout=0.0, weight_decay=0.1, start_weight_decay=None, end_weight_decay=None, weight_decay_incr_style='constant', clip_grad=1.0, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, micro_batch_size=1, batch_size=None, global_batch_size=16, rampup_batch_size=None, decrease_batch_size_if_needed=False, recompute_activations=False, recompute_granularity='selective', check_for_nan_in_loss_and_grad=True, check_for_spiky_loss=False, check_for_large_grads=False, distribute_saved_activations=False, recompute_method=None, recompute_num_layers=None, recompute_modules=['core_attn'], clone_scatter_output_in_embedding=True, profile=False, profile_step_start=10, profile_step_end=12, iterations_to_skip=[], result_rejected_tracker_filename=None, enable_gloo_process_groups=True, use_pytorch_profiler=False, profile_ranks=[0], record_memory_history=False, memory_snapshot_path='snapshot.pickle', tp_comm_overlap=False, tp_comm_overlap_cfg=None, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_bulk_dgrad=True, tp_comm_bulk_wgrad=True, tp_comm_bootstrap_backend='nccl', use_cpu_initialization=True, empty_unused_memory_level=0, deterministic_mode=False, check_weight_hash_across_dp_replicas_interval=None, calculate_per_token_loss=True, train_sync_interval=None, checkpoint_activations=False, train_iters=None, train_samples=None, log_interval=5, exit_interval=None, exit_duration_in_mins=None, exit_signal_handler=False, tensorboard_dir='/models/ZhipuAI/GLM-4.5-Air-Debug/runs', masked_softmax_fusion=True, bias_gelu_fusion=True, bias_swiglu_fusion=True, bias_dropout_fusion=True, apply_rope_fusion=True, cross_entropy_loss_fusion=False, cross_entropy_fusion_impl='native', use_flash_attn=False, add_bias_linear=False, add_qkv_bias=True, optimizer='adam', optimizer_cpu_offload=False, optimizer_offload_fraction=1.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, dataloader_type='cyclic', async_tensor_model_parallel_allreduce=True, no_persist_layer_norm=False, sequence_parallel=False, gradient_accumulation_fusion=True, deprecated_use_mcore_models=False, use_legacy_models=False, manual_gc=False, manual_gc_interval=0, manual_gc_eval=True, tp_comm_split_ag=True, tp_comm_split_rs=True, pipeline_model_parallel_comm_backend=None, high_priority_stream_groups=[], seed=42, data_parallel_random_init=False, init_method_std=0.02, init_method_xavier_uniform=False, lr=1e-05, lr_decay_style='cosine', lr_wsd_decay_style='exponential', lr_decay_iters=None, lr_decay_samples=None, lr_wsd_decay_samples=None, lr_wsd_decay_iters=None, lr_warmup_fraction=None, lr_warmup_iters=0, lr_warmup_samples=0, lr_warmup_init=0.0, warmup=None, min_lr=0.0, override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False, decoupled_lr=None, decoupled_min_lr=None, save='/models/ZhipuAI/GLM-4.5-Air-Debug', save_interval=500, no_save_optim=True, no_save_rng=True, load=None, no_load_optim=True, no_load_rng=True, non_persistent_save_interval=None, non_persistent_ckpt_type=None, non_persistent_global_ckpt_dir=None, non_persistent_local_ckpt_dir=None, non_persistent_local_ckpt_algo='fully_parallel', finetune=False, pretrained_checkpoint=None, ckpt_step=None, perform_initialization=False, use_checkpoint_args=False, use_mp_args_from_checkpoint_args=False, use_tokenizer_model_from_checkpoint_args=True, exit_on_missing_checkpoint=True, use_dist_ckpt_deprecated=False, use_persistent_ckpt_worker=False, auto_detect_ckpt_format=True, dist_ckpt_format_deprecated=None, ckpt_format='torch_dist', ckpt_convert_format=None, ckpt_convert_save=None, ckpt_convert_update_legacy_dist_opt_format=False, ckpt_fully_parallel_save_deprecated=False, ckpt_fully_parallel_save=True, async_save=None, ckpt_fully_parallel_load=False, ckpt_assume_constant_structure=False, dist_ckpt_strictness='assume_ok_unexpected', load_model_opt_format=False, fp16=False, bf16=True, grad_reduce_in_bf16=False, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, accumulate_allreduce_grads_in_fp32=False, fp16_lm_cross_entropy=False, disable_bf16_reduced_precision_matmul=False, reuse_grad_buf_for_mxfp8_param_ag=False, tensor_model_parallel_size=1, encoder_tensor_model_parallel_size=0, pipeline_model_parallel_size=1, encoder_pipeline_model_parallel_size=0, pipeline_model_parallel_split_rank=None, decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None, pipeline_model_parallel_layout=None, model_parallel_size=None, num_layers_per_virtual_pipeline_stage=None, num_virtual_stages_per_pipeline_rank=None, microbatch_group_size_per_vp_stage=None, overlap_p2p_comm=True, overlap_p2p_comm_warmup_flush=False, distributed_backend='nccl', distributed_timeout_minutes=300000, overlap_grad_reduce=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, align_grad_reduce=True, ddp_num_buckets=None, ddp_bucket_size=None, ddp_pad_buckets_for_high_nccl_busbw=False, ddp_average_in_collective=False, overlap_param_gather=False, overlap_param_gather_with_optimizer_step=False, align_param_gather=True, scatter_gather_tensors_in_pipeline=True, use_ring_exchange_p2p=False, local_rank=0, lazy_mpu_init=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, use_distributed_optimizer=True, nccl_ub=False, use_sharp=False, use_custom_fsdp=False, init_model_with_meta_device=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, fsdp_double_buffer=False, suggested_communication_unit_size=None, keep_fp8_transpose_cache_when_using_custom_fsdp=False, num_distributed_optimizer_instances=1, use_torch_fsdp2=False, torch_fsdp2_reshard_after_forward=True, context_parallel_size=1, cp_comm_type=['p2p'], hierarchical_context_parallel_sizes=None, nccl_communicator_config_path=None, use_tp_pp_dp_mapping=False, replication=False, replication_jump=None, replication_factor=2, eval_iters=-1, eval_interval=500, test_mode=False, skip_train=False, data_path=None, split=None, train_data_path=None, valid_data_path=None, test_data_path=None, data_args_path=None, per_split_data_args_path=None, data_cache_path=None, mmap_bin_files=True, mock_data=False, seq_length=131072, encoder_seq_length=None, decoder_seq_length=None, retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1, num_workers=4, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask_in_dataloader=True, num_dataset_builder_threads=1, object_storage_cache_path=None, mid_level_dataset_surplus=0.005, vocab_size=None, vocab_file=None, merge_file=None, vocab_extra_ids=0, tokenizer_type=None, tokenizer_model=None, tiktoken_pattern=None, tiktoken_num_special_tokens=1000, tiktoken_special_tokens=None, adlr_autoresume=False, adlr_autoresume_interval=1000, ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], retriever_score_scaling=False, block_data_path=None, embedding_path=None, indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, img_h=224, img_w=224, num_channels=3, patch_dim=16, classes_fraction=1.0, data_per_class_fraction=1.0, data_sharding=True, head_lr_mult=1.0, vision_pretraining=False, vision_pretraining_type='classify', vision_backbone_type='vit', swin_backbone_type='tiny', mask_type='random', mask_factor=1.0, iter_per_epoch=1250, dino_local_img_size=96, dino_local_crops_number=10, dino_head_hidden_size=2048, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_norm_last_layer=False, dino_warmup_teacher_temp=0.04, dino_teacher_temp=0.07, dino_warmup_teacher_temp_epochs=30, qk_layernorm=False, qk_l2_norm=False, expert_model_parallel_size=1, expert_tensor_parallel_size=None, num_experts=128, moe_layer_freq=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], moe_ffn_hidden_size=1408, moe_shared_expert_intermediate_size=1408, moe_shared_expert_overlap=False, moe_grouped_gemm=True, moe_use_legacy_grouped_gemm=False, moe_layer_recompute=False, moe_extended_tp=False, moe_use_upcycling=False, moe_router_load_balancing_type='aux_loss', moe_router_dtype='fp32', moe_router_score_function='sigmoid', moe_router_topk=8, moe_router_pre_softmax=False, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_topk_scaling_factor=1.0, moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001, moe_router_force_load_balancing=False, moe_router_padding_for_fp8=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_per_layer_logging=False, moe_token_dispatcher_type='alltoall', moe_enable_deepep=False, moe_deepep_num_sms=20, moe_permute_fusion=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_apply_probs_on_input=False, delay_wgrad_compute=False, moe_upcycling_granularity=1, q_lora_rank=None, kv_lora_rank=32, qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128, rotary_scaling_factor=1.0, mscale=1.0, mscale_all_dim=1.0, heterogeneous_layers_config_path=None, heterogeneous_layers_config_encoded_json=None, log_params_norm=False, log_num_zeros_in_grad=False, log_throughput=False, log_progress=False, timing_log_level=0, log_energy=False, barrier_with_L1_time=True, timing_log_option='minmax', tensorboard_log_interval=1, tensorboard_queue_size=50, log_timers_to_tensorboard=True, log_loss_scale_to_tensorboard=True, log_validation_ppl_to_tensorboard=True, log_memory_to_tensorboard=True, log_world_size_to_tensorboard=False, wandb_project='', wandb_exp_name='', wandb_save_dir='', logging_level=None, log_straggler=False, disable_straggler_on_startup=False, straggler_ctrlr_port=65535, straggler_minmax_count=1, run_workload_inspector_server=False, inference_batch_times_seqlen_threshold=-1, max_tokens_to_oom=12000, output_bert_embeddings=False, bert_embedder_type='megatron', flash_decode=False, enable_cuda_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, cuda_graph_scope='full', inference_max_batch_size=8, inference_max_seq_length=2560, inference_dynamic_batching=False, inference_dynamic_batching_buffer_size_gb=40.0, inference_dynamic_batching_chunk_size=256, inference_dynamic_batching_buffer_guaranteed_fraction=0.2, inference_dynamic_batching_buffer_overflow_factor=None, inference_dynamic_batching_max_requests_override=None, inference_dynamic_batching_max_tokens_override=None, symmetric_ar_type=None, nccl_all_reduce_for_prefill=False, mlp_chunks_for_prefill=1, fp8=None, fp8_recipe='delayed', fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1024, fp8_amax_compute_algo='max', fp8_wgrad=True, transformer_impl='transformer_engine', fp8_param_gather=False, first_last_layers_bf16=False, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1, te_rng_tracker=False, inference_rng_tracker=False, retro_project_dir=None, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_layers=2, retro_encoder_hidden_dropout=0.1, retro_encoder_attention_dropout=0.1, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_attention_gate=1, retro_verify_neighbor_count=True, enable_experimental=False, spec=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, hybrid_override_pattern=None, mamba_state_dim=128, mamba_head_dim=64, mamba_num_groups=8, mamba_num_heads=None, is_hybrid_model=False, disable_mamba_mem_eff_path=False, yaml_cfg=None, use_precision_aware_optimizer=False, main_grads_dtype='fp32', main_params_dtype='fp32', exp_avg_dtype='fp32', exp_avg_sq_dtype='fp32', enable_one_logger=True, one_logger_project='megatron-lm', one_logger_run_name=None, one_logger_async=False, app_tag_run_name=None, app_tag_run_version='0.0.0', inprocess_restart=False, inprocess_max_iterations=None, inprocess_monitor_thread_interval=1.0, inprocess_monitor_process_interval=1.0, inprocess_progress_watchdog_interval=1.0, inprocess_heartbeat_interval=30, inprocess_soft_timeout=60, inprocess_hard_timeout=90, inprocess_heartbeat_timeout=60, inprocess_barrier_timeout=120, inprocess_completion_timeout=120, inprocess_last_call_wait=1, inprocess_termination_grace_time=1, inprocess_granularity='node', inprocess_active_world_size=1, inprocess_empty_cuda_cache=False, enable_ft_package=False, calc_ft_timeouts=False, config_logger_dir='', error_injection_rate=0, error_injection_type='transient_error', rerun_mode='disabled', enable_msc=True, kitchen_config_file=None, kitchen_recipe_number=None, sft=False, sft_tokenizer_prompt_format='nemotron-h-aligned', rank=0, world_size=1)

加工后最终的 args (也是传入 model_provider 的 args):
Namespace(num_layers=46, encoder_num_layers=46, decoder_num_layers=None, hidden_size=4096, ffn_hidden_size=10944, num_attention_heads=96, attention_backend=<AttnBackend.unfused: 3>, kv_channels=128, group_query_attention=True, num_query_groups=8, max_position_embeddings=131072, position_embedding_type='rope', relative_attention_num_buckets=32, relative_attention_max_distance=128, use_rotary_position_embeddings=False, rotary_base=1000000, rotary_percent=1.0, rotary_interleaved=False, rotary_seq_len_interpolation_factor=None, use_rope_scaling=False, rope_scaling_factor=8.0, no_rope_freq=None, add_position_embedding=True, mrope_section=None, make_vocab_size_divisible_by=128, normalization='RMSNorm', norm_epsilon=1e-05, apply_layernorm_1p=False, apply_residual_connection_post_layernorm=False, openai_gelu=False, squared_relu=False, swiglu=True, onnx_safe=None, bert_binary_head=True, untie_embeddings_and_output_weights=True, multi_latent_attention=False, mtp_num_layers=None, mtp_loss_scaling_factor=0.1, attention_dropout=0.0, hidden_dropout=0.0, weight_decay=0.1, start_weight_decay=0.1, end_weight_decay=0.1, weight_decay_incr_style='constant', clip_grad=1.0, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, micro_batch_size=1, global_batch_size=16, rampup_batch_size=None, decrease_batch_size_if_needed=False, recompute_granularity='selective', check_for_nan_in_loss_and_grad=True, check_for_spiky_loss=False, check_for_large_grads=False, distribute_saved_activations=False, recompute_method=None, recompute_num_layers=None, recompute_modules=['core_attn'], clone_scatter_output_in_embedding=True, profile=False, profile_step_start=10, profile_step_end=12, iterations_to_skip=[], result_rejected_tracker_filename=None, enable_gloo_process_groups=True, use_pytorch_profiler=False, profile_ranks=[0], record_memory_history=False, memory_snapshot_path='snapshot.pickle', tp_comm_overlap=False, tp_comm_overlap_cfg=None, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_bulk_dgrad=True, tp_comm_bulk_wgrad=True, tp_comm_bootstrap_backend='nccl', use_cpu_initialization=True, empty_unused_memory_level=0, deterministic_mode=False, check_weight_hash_across_dp_replicas_interval=None, calculate_per_token_loss=True, train_sync_interval=None, train_iters=None, train_samples=None, log_interval=5, exit_interval=None, exit_duration_in_mins=None, exit_signal_handler=False, tensorboard_dir='/models/ZhipuAI/GLM-4.5-Air-Debug/runs', masked_softmax_fusion=True, bias_gelu_fusion=False, bias_swiglu_fusion=True, bias_dropout_fusion=True, apply_rope_fusion=True, cross_entropy_loss_fusion=False, cross_entropy_fusion_impl='native', use_flash_attn=False, add_bias_linear=False, add_qkv_bias=True, optimizer='adam', optimizer_cpu_offload=False, optimizer_offload_fraction=1.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, dataloader_type='cyclic', async_tensor_model_parallel_allreduce=True, no_persist_layer_norm=False, sequence_parallel=False, gradient_accumulation_fusion=True, deprecated_use_mcore_models=False, use_legacy_models=False, manual_gc=False, manual_gc_interval=0, manual_gc_eval=True, tp_comm_split_ag=True, tp_comm_split_rs=True, pipeline_model_parallel_comm_backend=None, high_priority_stream_groups=[], seed=42, data_parallel_random_init=False, init_method_std=0.02, init_method_xavier_uniform=False, lr=1e-05, lr_decay_style='cosine', lr_wsd_decay_style='exponential', lr_decay_iters=None, lr_decay_samples=None, lr_wsd_decay_samples=None, lr_wsd_decay_iters=None, lr_warmup_fraction=None, lr_warmup_iters=0, lr_warmup_samples=0, lr_warmup_init=0.0, min_lr=0.0, override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False, decoupled_lr=None, decoupled_min_lr=None, save='/models/ZhipuAI/GLM-4.5-Air-Debug', save_interval=500, no_save_optim=True, no_save_rng=True, load=None, no_load_optim=True, no_load_rng=True, non_persistent_save_interval=None, non_persistent_ckpt_type=None, non_persistent_global_ckpt_dir=None, non_persistent_local_ckpt_dir=None, non_persistent_local_ckpt_algo='fully_parallel', finetune=False, pretrained_checkpoint=None, ckpt_step=None, perform_initialization=False, use_checkpoint_args=False, use_mp_args_from_checkpoint_args=False, use_tokenizer_model_from_checkpoint_args=True, exit_on_missing_checkpoint=True, use_dist_ckpt_deprecated=False, use_persistent_ckpt_worker=False, auto_detect_ckpt_format=True, dist_ckpt_format_deprecated=None, ckpt_format='torch_dist', ckpt_convert_format=None, ckpt_convert_save=None, ckpt_convert_update_legacy_dist_opt_format=False, ckpt_fully_parallel_save_deprecated=False, ckpt_fully_parallel_save=True, async_save=None, ckpt_fully_parallel_load=False, ckpt_assume_constant_structure=False, dist_ckpt_strictness='assume_ok_unexpected', load_model_opt_format=False, fp16=False, bf16=True, grad_reduce_in_bf16=False, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, accumulate_allreduce_grads_in_fp32=True, fp16_lm_cross_entropy=False, disable_bf16_reduced_precision_matmul=False, reuse_grad_buf_for_mxfp8_param_ag=False, tensor_model_parallel_size=1, encoder_tensor_model_parallel_size=0, pipeline_model_parallel_size=1, encoder_pipeline_model_parallel_size=0, pipeline_model_parallel_split_rank=None, decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None, pipeline_model_parallel_layout=None, num_layers_per_virtual_pipeline_stage=None, num_virtual_stages_per_pipeline_rank=None, microbatch_group_size_per_vp_stage=None, overlap_p2p_comm=False, overlap_p2p_comm_warmup_flush=False, distributed_backend='nccl', distributed_timeout_minutes=300000, overlap_grad_reduce=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, align_grad_reduce=True, ddp_num_buckets=None, ddp_bucket_size=None, ddp_pad_buckets_for_high_nccl_busbw=False, ddp_average_in_collective=False, overlap_param_gather=False, overlap_param_gather_with_optimizer_step=False, align_param_gather=False, scatter_gather_tensors_in_pipeline=True, use_ring_exchange_p2p=False, local_rank=0, lazy_mpu_init=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, use_distributed_optimizer=True, nccl_ub=False, use_sharp=False, use_custom_fsdp=False, init_model_with_meta_device=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, fsdp_double_buffer=False, suggested_communication_unit_size=None, keep_fp8_transpose_cache_when_using_custom_fsdp=False, num_distributed_optimizer_instances=1, use_torch_fsdp2=False, torch_fsdp2_reshard_after_forward=True, context_parallel_size=1, cp_comm_type=['p2p'], hierarchical_context_parallel_sizes=None, nccl_communicator_config_path=None, use_tp_pp_dp_mapping=False, replication=False, replication_jump=None, replication_factor=2, eval_iters=-1, eval_interval=500, test_mode=False, skip_train=False, data_path=None, split=None, train_data_path=None, valid_data_path=None, test_data_path=None, data_args_path=None, per_split_data_args_path=None, data_cache_path=None, mmap_bin_files=True, mock_data=False, seq_length=131072, encoder_seq_length=131072, decoder_seq_length=None, retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1, num_workers=4, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask_in_dataloader=True, num_dataset_builder_threads=1, object_storage_cache_path=None, mid_level_dataset_surplus=0.005, vocab_size=None, vocab_file=None, merge_file=None, vocab_extra_ids=0, tokenizer_type=None, tokenizer_model=None, tiktoken_pattern=None, tiktoken_num_special_tokens=1000, tiktoken_special_tokens=None, adlr_autoresume=False, adlr_autoresume_interval=1000, ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], retriever_score_scaling=False, block_data_path=None, embedding_path=None, indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, img_h=224, img_w=224, num_channels=3, patch_dim=16, classes_fraction=1.0, data_per_class_fraction=1.0, data_sharding=True, head_lr_mult=1.0, vision_pretraining=False, vision_pretraining_type='classify', vision_backbone_type='vit', swin_backbone_type='tiny', mask_type='random', mask_factor=1.0, iter_per_epoch=1250, dino_local_img_size=96, dino_local_crops_number=10, dino_head_hidden_size=2048, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_norm_last_layer=False, dino_warmup_teacher_temp=0.04, dino_teacher_temp=0.07, dino_warmup_teacher_temp_epochs=30, qk_layernorm=False, qk_l2_norm=False, expert_model_parallel_size=1, expert_tensor_parallel_size=1, num_experts=128, moe_layer_freq=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], moe_ffn_hidden_size=1408, moe_shared_expert_intermediate_size=1408, moe_shared_expert_overlap=False, moe_grouped_gemm=True, moe_use_legacy_grouped_gemm=False, moe_layer_recompute=False, moe_extended_tp=False, moe_use_upcycling=False, moe_router_load_balancing_type='aux_loss', moe_router_dtype='fp32', moe_router_score_function='sigmoid', moe_router_topk=8, moe_router_pre_softmax=False, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_topk_scaling_factor=1.0, moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001, moe_router_force_load_balancing=False, moe_router_padding_for_fp8=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_per_layer_logging=False, moe_token_dispatcher_type='alltoall', moe_enable_deepep=False, moe_deepep_num_sms=20, moe_permute_fusion=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_apply_probs_on_input=False, delay_wgrad_compute=False, moe_upcycling_granularity=1, q_lora_rank=None, kv_lora_rank=32, qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128, rotary_scaling_factor=1.0, mscale=1.0, mscale_all_dim=1.0, heterogeneous_layers_config_path=None, heterogeneous_layers_config_encoded_json=None, log_params_norm=False, log_num_zeros_in_grad=False, log_throughput=False, log_progress=False, timing_log_level=0, log_energy=False, barrier_with_L1_time=True, timing_log_option='minmax', tensorboard_log_interval=1, tensorboard_queue_size=50, log_timers_to_tensorboard=True, log_loss_scale_to_tensorboard=True, log_validation_ppl_to_tensorboard=True, log_memory_to_tensorboard=True, log_world_size_to_tensorboard=False, wandb_project='', wandb_exp_name='', wandb_save_dir='', logging_level=None, log_straggler=False, disable_straggler_on_startup=False, straggler_ctrlr_port=65535, straggler_minmax_count=1, run_workload_inspector_server=False, inference_batch_times_seqlen_threshold=-1, max_tokens_to_oom=12000, output_bert_embeddings=False, bert_embedder_type='megatron', flash_decode=False, enable_cuda_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, cuda_graph_scope='full', inference_max_batch_size=8, inference_max_seq_length=2560, inference_dynamic_batching=False, inference_dynamic_batching_buffer_size_gb=40.0, inference_dynamic_batching_chunk_size=256, inference_dynamic_batching_buffer_guaranteed_fraction=0.2, inference_dynamic_batching_buffer_overflow_factor=None, inference_dynamic_batching_max_requests_override=None, inference_dynamic_batching_max_tokens_override=None, symmetric_ar_type=None, nccl_all_reduce_for_prefill=False, mlp_chunks_for_prefill=1, fp8=None, fp8_recipe='delayed', fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1024, fp8_amax_compute_algo='max', fp8_wgrad=True, transformer_impl='transformer_engine', fp8_param_gather=False, first_last_layers_bf16=False, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1, te_rng_tracker=False, inference_rng_tracker=False, retro_project_dir=None, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_layers=2, retro_encoder_hidden_dropout=0.1, retro_encoder_attention_dropout=0.1, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_attention_gate=1, retro_verify_neighbor_count=True, enable_experimental=False, spec=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, hybrid_override_pattern=None, mamba_state_dim=128, mamba_head_dim=64, mamba_num_groups=8, mamba_num_heads=None, is_hybrid_model=False, disable_mamba_mem_eff_path=False, yaml_cfg=None, use_precision_aware_optimizer=False, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, enable_one_logger=True, one_logger_project='megatron-lm', one_logger_run_name=None, one_logger_async=False, app_tag_run_name=None, app_tag_run_version='0.0.0', inprocess_restart=False, inprocess_max_iterations=None, inprocess_monitor_thread_interval=1.0, inprocess_monitor_process_interval=1.0, inprocess_progress_watchdog_interval=1.0, inprocess_heartbeat_interval=30, inprocess_soft_timeout=60, inprocess_hard_timeout=90, inprocess_heartbeat_timeout=60, inprocess_barrier_timeout=120, inprocess_completion_timeout=120, inprocess_last_call_wait=1, inprocess_termination_grace_time=1, inprocess_granularity='node', inprocess_active_world_size=1, inprocess_empty_cuda_cache=False, enable_ft_package=False, calc_ft_timeouts=False, config_logger_dir='', error_injection_rate=0, error_injection_type='transient_error', rerun_mode='disabled', enable_msc=True, kitchen_config_file=None, kitchen_recipe_number=None, sft=False, sft_tokenizer_prompt_format='nemotron-h-aligned', rank=0, world_size=1, use_dist_ckpt=True, transformer_pipeline_model_parallel_size=1, data_parallel_size=1, train_type='full', freeze_parameters=[], freeze_parameters_regex=None, freeze_parameters_ratio=0.0, trainable_parameters=[], trainable_parameters_regex=None, adapter_load=None, target_modules=['all-linear'], target_regex=None, modules_to_save=[], lora_rank=8, lora_alpha=32, lora_dropout=0.05, lora_bias='none', lora_dtype=None, use_rslora=False, ref_load=None, beta=0.1, rpo_alpha=None, reference_free=False, label_smoothing=0.0, f_divergence_type='reverse_kl', loss_type='sigmoid', padded_vocab_size=151552, initialize_embedding=False, rope_scaling=None, torch_dtype=torch.bfloat16, padding_free=True, mlp_padding_free=False, dataloader_persistent_workers=True, dataloader_prefetch_factor=10, architectures='Glm4MoeForCausalLM', max_epochs=None, enable_dft_loss=False, original_max_position_embeddings=None, partial_rotary_factor=0.5, use_shared_expert_gate=False, virtual_pipeline_model_parallel_size=None, params_dtype=torch.bfloat16, consumed_train_samples=0, skipped_train_samples=0, consumed_valid_samples=0, variable_seq_lengths=False)


### config
TransformerConfig(tensor_model_parallel_size=1, pipeline_model_parallel_comm_backend=None, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=False, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=1, moe_extended_tp=False, perform_initialization=False, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=True, async_tensor_model_parallel_allreduce=True, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, cross_entropy_fusion_impl='native', tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=True, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, delay_wgrad_compute=False, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=46, mtp_num_layers=None, mtp_loss_scaling_factor=0.1, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, pipeline_model_parallel_layout=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=4096, num_attention_heads=96, attention_backend=<AttnBackend.unfused: 3>, softmax_scale=None, num_query_groups=8, ffn_hidden_size=10944, kv_channels=128, hidden_dropout=0.0, attention_dropout=0.0, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=True, gated_linear_unit=True, activation_func=<function silu at 0x7fed6ee89120>, activation_func_fp8_input_store=False, num_moe_experts=128, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=True, multi_latent_attention=False, no_rope_freq=None, moe_deepep_num_sms=20, init_method=functools.partial(<function normal_ at 0x7fed6ed13100>, mean=0.0, std=0.02), output_layer_init_method=functools.partial(<function normal_ at 0x7fed6ed13100>, mean=0.0, std=0.002085144140570748), init_method_std=0.02, init_model_with_meta_device=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, disable_bf16_reduced_precision_matmul=False, bias_activation_fusion=True, masked_softmax_fusion=True, persist_layer_norm=True, memory_efficient_layer_norm=False, bias_dropout_fusion=True, apply_rope_fusion=True, recompute_granularity='selective', recompute_method=None, recompute_num_layers=None, distribute_saved_activations=False, recompute_modules=['core_attn'], fp8=None, fp8_recipe='delayed', fp8_param=False, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1024, fp8_amax_compute_algo='max', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, first_last_layers_bf16=False, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1, use_kitchen=False, moe_shared_expert_intermediate_size=1408, moe_shared_expert_overlap=False, moe_layer_freq=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], moe_ffn_hidden_size=1408, moe_router_load_balancing_type='aux_loss', moe_router_topk=8, moe_router_topk_limited_devices=None, moe_router_padding_for_fp8=False, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=1.0, moe_router_score_function='sigmoid', moe_router_dtype='fp32', moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001, moe_router_force_load_balancing=False, moe_grouped_gemm=True, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_enable_deepep=False, moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, moe_apply_probs_on_input=False, cp_comm_type='p2p', enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, cuda_graph_scope='full', clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False, symmetric_ar_type=None, mrope_section=None, is_hybrid_model=False, mamba_state_dim=128, mamba_head_dim=64, mamba_num_groups=8, mamba_num_heads=None, use_mamba_mem_eff_path=True, mlp_chunks_for_prefill=1, heterogeneous_block_specs=False, hetereogenous_dist_checkpoint=False, quant_recipe=None)


### hf_model
Glm4MoeForCausalLM(
  (model): Glm4MoeModel(
    (embed_tokens): Embedding(151552, 4096, padding_idx=151329)
    (layers): ModuleList(
      (0): Glm4MoeDecoderLayer(
        (self_attn): Glm4MoeAttention(
          (q_proj): Linear(in_features=4096, out_features=12288, bias=True)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (o_proj): Linear(in_features=12288, out_features=4096, bias=False)
        )
        (mlp): Glm4MoeMLP(
          (gate_proj): Linear(in_features=4096, out_features=10944, bias=False)
          (up_proj): Linear(in_features=4096, out_features=10944, bias=False)
          (down_proj): Linear(in_features=10944, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Glm4MoeRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): Glm4MoeRMSNorm((4096,), eps=1e-05)
      )
      (1-45): 45 x Glm4MoeDecoderLayer(
        (self_attn): Glm4MoeAttention(
          (q_proj): Linear(in_features=4096, out_features=12288, bias=True)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (o_proj): Linear(in_features=12288, out_features=4096, bias=False)
        )
        (mlp): Glm4MoeMoE(
          (experts): ModuleList(
            (0-127): 128 x Glm4MoeMLP(
              (gate_proj): Linear(in_features=4096, out_features=1408, bias=False)
              (up_proj): Linear(in_features=4096, out_features=1408, bias=False)
              (down_proj): Linear(in_features=1408, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
          )
          (gate): Glm4MoeTopkRouter()
          (shared_experts): Glm4MoeMLP(
            (gate_proj): Linear(in_features=4096, out_features=1408, bias=False)
            (up_proj): Linear(in_features=4096, out_features=1408, bias=False)
            (down_proj): Linear(in_features=1408, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
        )
        (input_layernorm): Glm4MoeRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): Glm4MoeRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): Glm4MoeRMSNorm((4096,), eps=1e-05)
    (rotary_emb): Glm4MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=151552, bias=False)
)


### mg_model
GPTModel(
  (embedding): LanguageModelEmbedding(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (rotary_pos_emb): RotaryEmbedding()
  (decoder): TransformerBlock(
    (layers): ModuleList(
      (0): TransformerLayer(
        (input_layernorm): IdentityOp()
        (self_attention): SelfAttention(
          (core_attention): TEDotProductAttention(
            (flash_attention): FlashAttention()
            (fused_attention): FusedAttention()
            (unfused_attention): UnfusedDotProductAttention(
              (scale_mask_softmax): FusedScaleMaskSoftmax()
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (linear_proj): TERowParallelLinear(in_features=12288, out_features=4096, bias=False, TP=1)
          (linear_qkv): TELayerNormColumnParallelLinear(in_features=4096, out_features=14336, bias=True, TP=1)
          (q_layernorm): IdentityOp()
          (k_layernorm): IdentityOp()
        )
        (pre_cross_attn_layernorm): IdentityOp()
        (cross_attention): IdentityOp()
        (cross_attn_bda): IdentityFuncOp()
        (pre_mlp_layernorm): IdentityOp()
        (mlp): MLP(
          (linear_fc1): TELayerNormColumnParallelLinear(in_features=4096, out_features=21888, bias=False, TP=1)
          (linear_fc2): TERowParallelLinear(in_features=10944, out_features=4096, bias=False, TP=1)
        )
      )
      (1-45): 45 x TransformerLayer(
        (input_layernorm): IdentityOp()
        (self_attention): SelfAttention(
          (core_attention): TEDotProductAttention(
            (flash_attention): FlashAttention()
            (fused_attention): FusedAttention()
            (unfused_attention): UnfusedDotProductAttention(
              (scale_mask_softmax): FusedScaleMaskSoftmax()
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (linear_proj): TERowParallelLinear(in_features=12288, out_features=4096, bias=False, TP=1)
          (linear_qkv): TELayerNormColumnParallelLinear(in_features=4096, out_features=14336, bias=True, TP=1)
          (q_layernorm): IdentityOp()
          (k_layernorm): IdentityOp()
        )
        (pre_cross_attn_layernorm): IdentityOp()
        (cross_attention): IdentityOp()
        (cross_attn_bda): IdentityFuncOp()
        (pre_mlp_layernorm): RMSNorm()
        (mlp): MoELayer(
          (router): TopKRouter()
          (experts): TEGroupedMLP(
            (linear_fc1): TEColumnParallelGroupedLinear()
            (linear_fc2): TERowParallelGroupedLinear()
          )
          (shared_experts): SharedExpertMLP(
            (linear_fc1): TEColumnParallelLinear(in_features=4096, out_features=2816, bias=False, TP=1)
            (linear_fc2): TERowParallelLinear(in_features=1408, out_features=4096, bias=False, TP=1)
          )
        )
      )
    )
    (final_layernorm): RMSNorm()
  )
  (output_layer): ColumnParallelLinear(in_features=4096, out_features=151552, bias=False, TP=1)
)
