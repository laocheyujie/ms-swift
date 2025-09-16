## Megatron
swift.cli._megatron.sft.py -> 
megatron_sft_main() ->
MegatronSft(args).main() ->
result = self.run() ->
args = self.args
train_dataset, val_dataset = self._prepare_dataset()
data_collator = self._get_data_collator()
self.trainer.train(train_dataset, val_dataset, data_collator) ->
datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)
pretrain(datasets_provider, model_provider, model_type, forward_step_func, ..., extra_args_provider, args_defaults) ->
initialize_megatron
set_jit_fusion_options()
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider_func, model_type, checkpointing_context={}) ->
model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(new_model_provider_func, model_type, *_args, **kwargs) ->
model = get_model(model_provider_func, model_type) ->
model = build_model() ->
model = model_provider_func(pre_process=pre_process, post_process=post_process) ->
实际进入了 swift.megatron.trainers.base.py new_model_provider_func ->
self.unwrapped_model = model_provider_func(*args, **kwargs) ->
transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(...) ->


model_provider_func -> 
get_gpt_decoder_block_spec -> 
get_gpt_layer_with_transformer_engine_spec 得到 dense_layer_spec 和 moe_layer_spec，遍历 num_layers，根据 moe_layer_pattern 来堆叠 layer_specs 列表里每个 layer，获取当前 pp 的 local_layer_specs，组装成 block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl) -> 
selfattention 模块是通用的，mlp 要区分是 dense 还是 moe，mlp = get_mlp_module_spec_for_backend(...) 得到 mlp 层，组装 ModuleSpec(...) ->
Dense: linear_fc1: TELayerNormColumnParallelLinear; linear_fc2: TERowParallelLinear，组装成 mlp = ModuleSpec(module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2))
MoE(get_moe_module_spec_for_backend): 
    - 共享专家：linear_fc1: TEColumnParallelLinear; linear_fc2: TERowParallelLinear，组装成共享专家的子层 mlp = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)，shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)
    - 普通专家：linear_fc1: TEColumnParallelGroupedLinear; linear_fc2: TERowParallelGroupedLinear，组成普通专家的子层 expert_submodule = MLPSubmodules(linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear)，experts = ModuleSpec(module=TEGroupedMLP, submodules=expert_submodule)
组成 MoE: moe_module_spec = ModuleSpec(module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts))

    



TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear), 在 __init__ 方法里调用了父类的 __init__, 父类里有:
```py
if self.parallel_mode == "column":
    self.out_features = divide(self.out_features, self.tp_size)
elif self.parallel_mode == "row":
    self.in_features = divide(self.in_features, self.tp_size)
```


args.finetune or release: iteration = 0，否则会从 state_dict 里读取 iteration


step_batch_size = args.micro_batch_size * data_parallel_size 即 micro_batch_size 决定多少个数据进行一次参数更新



## 简版流程
1. 初始化 megatron 运行环境 `init_megatron_env` 
    1. 安装：确认安装 `Megatron-LM`
    2. 打补丁：对 Megatron 打一系列补丁 `_patch_megatron`
    3. 在导包过程中完成了模板注册 `register_template`，模型注册 `register_model_arch`, `register_model`, `register_megatron_model`，数据集注册 `register_dataset`, `register_dataset_info`
2. `MegatronSft(args)` 实例化
    1. 参数解析：使用父类初始化方法解析参数 `super(SwiftSft, self).__init__(args)`
    2. Processor：获取 processor 或 tokenizer `_, self.processor = args.get_model_processor(load_model=False)`
    3. Tokenizer 补丁：用 hf 的 tokenizer 代替 Megatron 定义的 tokenizer 方法 `patch_megatron_tokenizer`
    4. 参数转换：将 hf 的 config 转换为 Megatron 的参数 `convert_hf_config` (`swift.megatron.model.config.py`)
    5. Template：根据 `template_type` 准备对话模板 `self._prepare_template()` 并设置 `self.template.use_megatron = True`
    6. 保存参数：`args.save_args(args.save)` 存储转换后的相关参数
    7. 打 Megatron 训练补丁： `_patch_megatron`
        1. train_step: `training.train_step = self.train_step` 使其支持 max_epochs
        2. cyclic_iter: `training.cyclic_iter = self.new_cyclic_iter` 使其支持基于 epoch 的训练模式
        3. evaluate: `training.evaluate = self.evaluate`
        4. setup_model_and_optimizer: `training.setup_model_and_optimizer = self.setup_model_and_optimizer` 支持 LoRA 层的植入和权重读取
        5. save_checkpoint: `training.save_checkpoint = self.save_checkpoint` 只保存训练参数 (required_grad = True) 的部分
3. `MegatronSft(args).run()` 正式开始处理
    1. Dataset：`train_dataset, val_dataset = self._prepare_dataset()`
        1. 获取数据集：使用 `datasets.load_dataset` 加载数据集
        2. 数据预处理：`swift.llm.dataset.preprocessor.core.AutoPreprocessor.__call__(...)`
            1. Preprocessor：使用 `preprocessor = self._get_preprocessor(dataset)` 获取数据集对应的数据预处理器 (`MessagesPreprocessor`)
            2. 数据批量预处理：`RowPreprocessor.__call__(...)`
        2. 数据后处理：`train_dataset, val_dataset = DatasetLoader.post_process(...)`
            1. 计算验证集数量
            2. 切分数据集：`train_dataset, val_dataset = train_dataset.train_test_split(...)`
            3. 采样或洗牌数据集：`train_dataset = sample_dataset(...)`
        3. Encode：`train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)`
            1. 获取模板 template
            2. 保存验证集
            3. 实例化 `preprocessor = EncodePreprocessor(...)`
            4. 批处理：`RowPreprocessor.batched_preprocess`
                1. Encode：`Template.encode`
                    1. 规范化 tool
                    2. 获取 agent_template：`agent_template = self.agent_template`
                    3. 获取 template_meta`agent_template.template_meta = self.template_meta` 获取 template_meta
                    4. Encode：`encoded = self._encode(inputs)`
                        1. 对连续的 user 或 assistant 或 tool 进行 content 合并
                        2. 获取 system 内容：`system = self._get_system(inputs)`
                        3. 把 tools 放入 system：`system = self.agent_template._format_tools(tools, system or '', inputs.messages[0])`
                        4. Tokenize：`token_list = self._tokenize(context)`
                        5. Label: query 部分填充为 -100，response 保持原样
                        6. 把 `input_ids` 填充到 `cp_size * 2` 的整数倍
                2. Trunc：对于超过 `--max_length` 的按照 `self.truncation_strategy` 策略来截断
        4. Concat：把所有训练数据集子集都拼接到一起
        5. Packing：`dataset = packing_dataset_cls(...)` 
    2. Data Collator：`data_collator = self._get_data_collator()`
        1. 获取 data_collator：`data_collator = self.template.data_collator`
        2. 设置 Padding
    3. Train：`self.trainer.train(train_dataset, val_dataset, data_collator)` 开始训练
        1. datasets_provider：`datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)`
        2. 打补丁：对 `training.build_pretraining_data_loader` 打补丁
        3. iters：对 `training.initialize_megatron` 打补丁以兼容 `max_epochs` 的 `train_iters` 数
        4. model_provider：使用的是 `swift.megatron.model.gpt.model.py` 里的 `model_provider` 相较于 Megatron-LM 原版的 `model_provider`，做了如下变化：
            1. 去掉了 `ModelOpt` 的内容
            2. 去掉了 `vp_stage` 的内容
            3. 去掉了 `qk_l2_norm` 的内容
            4. 去掉了 `use_kitchen`
            5. 新增兼容 qwen2_moe 的 `shared_experts` 内容
        5. Pretrain：
            1. 初始化 Megatron 环境：`initialize_megatron(...)`
                1. 验证参数：`validate_args(args, args_defaults)`
                2. 设置全局变量：`set_global_variables(args)`
                3. 设置日志：`setup_logging()`
                4. 设置随机信息：`initialize_rerun_state_machine(...)`
                5. 分布式：`finish_mpu_init()` 主要通过 `mpu.initialize_model_parallel(...)` 设置模型并行，数据并行等各种进程组，每个 rank 对应进程都有自己全局变量
                6. 设置自动恢复：`_init_autoresume()`
            2. 编译融合算子：`set_jit_fusion_options`
            3. 核心三要素（模型、优化器、优化器调度器）：`model, optimizer, opt_param_scheduler = setup_model_and_optimizer(...)`
                1. 替换原有的 `setup_model_and_optimizer`
                2. 打高效 cat 补丁
                3. 做 LoRA key 映射
                4. `model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(new_model_provider_func, ...)`
                    1. model: `model = get_model(model_provider_func, model_type)`
                        1. base model: `self.unwrapped_model = model_provider_func(*args, **kwargs)`
                        2. peft model: `self.peft_model = prepare_mcore_model(self.unwrapped_model)`
                    2. ddp: `ddp_config = DistributedDataParallelConfig(**kwargs)` 根据 kwargs DDP 生成分布式配置实例
                    3. optimizer: `optimizer = get_megatron_optimizer(...)`
                    4. opt_param_scheduler: `opt_param_scheduler = get_optimizer_param_scheduler(optimizer)`
                    5. 加载基础模型权重：`load_checkpoint(model, optimizer, opt_param_scheduler, ...)`
            4. train_dataloader: `train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)`
                1. `train_ds, valid_ds, test_ds = build_train_valid_test_datasets(build_train_valid_test_datasets_provider)`
                2. `train_dataloader = build_pretraining_data_loader(train_ds, args.consumed_train_samples)`
                3. `valid_dataloader`, `test_dataloader` 同理
            5. data_iterator: 
                1. `train_data_iterator = _get_iterator(dl_type, train_dataloader)`
                2. `valid_data_iterator`, `test_data_iterator` 同理
            6. train: `iteration, num_floating_point_operations_so_far = train(...)` 开始训练
                1. model: 取出 model 里的 model_module `model_module.train()`
                2. train: `train_step(forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config)`







## 流程

1. `swift.cli._megatron.sft.py` -> `from swift.megatron import megatron_sft_main` -> `swift.megatron.__init__.py` 会进行导包初始化
    1. `init_megatron_env` 根据 `MEGATRON_LM_PATH` 来安装或者下载 megatron；`_patch_megatron` 对 Megatron 打一些补丁
        1. `from swift.megatron import tuners` patch LoRA -> ... -> `swift.tuners.peft.py` 里 `hot_patch_peft_module()` -> `LoraModel.__init__ = __new_init__`
    2. ...
    3. 注册数据集：`swift.llm.argument.base_args.data_args.py` -> `from .data_args import DataArguments` -> `swift.llm.dataset.dataset.llm.py` -> **`register_dataset`** 在注册的过程中会用到各种 `Preprocessor()` 的初始化，它们的基类是 `RowPreprocessor`
    4. 注册模型：`swift.megatron.argument.rlhf_args.py` -> `swift.megatron.argument.train_args.py` -> **`register_megatron_model`**
        1. 从 `MODEL_MAPPING` 拿到具体模型信息 `model_meta`
        2. `model_meta.support_megatron = True`
        3. 把模型加入到 `MEGATRON_MODEL_MAPPING`
2. `megatron_sft_main()` -> `MegatronSft(args).main()`
    1. 实例化调用 `MegatronSft.__init__()`
        1. `super(SwiftSft, self).__init__(args)`
            1. `SwiftPipeline.__init__`
                1. `self.args = self._parse_args(args)` -> `(self.args_class, args)`
                    1. 使用 `HfArgumentParser` 生成 `class_type`(`<class 'swift.megatron.argument.train_args.MegatronTrainArguments'>`) 的参数解析器
                    2. 从 json 参数配置或**命令行参数**里解析参数为 `argv`
                    3. 解析参数 `args, remaining_args = parser.parse_args_into_dataclasses(argv, return_remaining_strings=True)`
                    4. `self.args = args` 得到解析后的参数
                2. `MegatronTrainArguments.__post_init__` -> `BaseArguments.__post_init__(self)` -> `ModelArguments.__post_init__(self)` -> `ModelArguments._init_torch_dtype()` -> `ModelArguments._init_model_info()` -> `get_model_info_meta(**self.get_model_kwargs())`
                    1. `model_info = _get_model_info(...)`
                    2. `model_meta = MODEL_MAPPING[model_type]`
        2. `_, self.processor = args.get_model_processor(load_model=False)` 获取 processor 或 tokenizer
            1. `get_model_tokenizer(**kwargs)`
                1. `model_info, model_meta = get_model_info_meta()`
                    1. `model_info = _get_model_info(...)`
                    2. `model_meta = MODEL_MAPPING[model_type]`
                2. `device_map = get_default_device_map()`
                    1. 如果是 mp, `device_map = 'auto'`
                    2. 否则 `device_map = 'cuda:x'`
                3. `model, processor = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)` 实际调用的是 `get_model_tokenizer_with_flash_attn`
                    1. `model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)`
                    2. `return get_model_tokenizer_from_local(...)`
                        1. `tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)`
                        2. `model = None`
                        3. `return model, tokenizer`
                4. `tokenizer = processor`
                5. `new_special_tokens` 处理
                6. `return model=None, tokenizer`
        3. `patch_megatron_tokenizer` 用 hf 的 tokenizer 代替 Megatron 定义的 tokenizer 方法
        4. `args.init_model_args(self.processor, self.processor.model_info.config)`
            1. `self.megatron_model_meta = get_megatron_model_meta(self.model_type)` 得到 `MEGATRON_MODEL_MAPPING[_MODEL_META_MAPPING[model_type]]`
            2. `kwargs = self.megatron_model_meta.convert_hf_config(config)` 实际调用了 `convert_gpt_hf_config` 
                1. `convert_hf_config` 将 hf 的 config 转换为 Megatron 的参数 (**`swift.megatron.model.config.py` 包含 hf_config -> megatron_config 映射关系和转换函数**)
                2. 针对不同的模型做一些参数上的替换
            3. 把 Megatron 的参数设置为 `MegatronTrainArguments` 类属性
            4. `MegatronArguments.__post_init__(self)`
                1. `MegatronTunerMixin.__post_init__(self)` 做一点 freeze 和 pipline 确认, lora 的参数确认
                2. `self._set_default()` 设置默认参数
                3. `self._init_moe()` 对 MoE 模型的参数打一些补丁
                4. `self._init_mixed_precision()` 对混合精度做一些补丁
                    1. `ModelArguments._init_mixed_precision(self)` 设置一下 self.fp16 和 self.bf16
                5. `self.extra_megatron_kwargs = json_parse_to_dict(self.extra_megatron_kwargs)` 拿到 `extra_megatron_kwargs` 非 Megatron 标准参数
            5. `self.extra_args = self.parse_to_megatron()`
                1. `new_args, extra_args = self._args_to_argv()`
                    1. 把 Megatron 的标准参数转换命令函参数的形式 --xxx xxx
                    2. 把非 Megatron 标准参数提取出来
                2. 返回非 Megatron 标准参数 `extra_args`
        5. `self._prepare_template()` 准备对话模板
            1. `template = self.args.get_template(self.processor)`
                1. `template_kwargs = self.get_template_kwargs()` 返回 max_length, padding 策略等信息
                2. `template = get_template(template_type, processor, **template_kwargs)` 根据 `template_type` 获取 template 的类 <swift.llm.template.template.utils.ThinkingTemplate object at 0x7f1acdfdf9d0>
            2. `template.set_mode('train')`
            3. `args.save_args(args.save)` 在 master (rank=0 或 -1) 节点上持久化保存参数
            4. `self.trainer = self.prepare_trainer()` -> `return MegatronTrainer(self.args)` 实例化 `MegatronTrainer`
                1. `BaseMegatronTrainer._patch_megatron()` 
                    1. `training.train_step = self.train_step` 加工一下使其支持 max_epochs
                    2. `training.cyclic_iter = self.new_cyclic_iter` 支持基于 epoch 的训练模式
                    3. `training.evaluate = self.evaluate` 加工一下 evaluation
                    4. `training.setup_model_and_optimizer = self.setup_model_and_optimizer`
                    5. `training.save_checkpoint = self.save_checkpoint`
                        1. `with adapter_state_dict_context()` with 装饰器的作用是使得 `training.save_checkpoint` 时只保存训练参数 (required_grad = True) 的部分
    2. `result = self.run()` 正式开始处理
        1. dataset: `train_dataset, val_dataset = self._prepare_dataset()`
            1. `train_dataset, val_dataset = self._get_dataset()`
                1. `dataset_kwargs = args.get_dataset_kwargs()` 获取 dataset 相关的参数
                2. `train_dataset, val_dataset = load_dataset(...)` 获取训练数据和验证数据
                    1. 初始化自我认知数据 `init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)` 把数据集里的 name 和 author 替换成 `model_name, model_author`
                    2. `dataset_syntax = DatasetSyntax.parse(dataset)`
                    3. `dataset_meta = dataset_syntax.get_dataset_meta(use_hf)`
                    4. `train_dataset = load_function(dataset_syntax, dataset_meta, **load_kwargs, use_hf=use_hf)` 实际调用的是 `DatasetLoader.load`
                        1. `dataset = DatasetLoader._load_dataset_path(...)`
                            1. 获取文件类型 `file_type`
                            2. `dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)` 实际使用的是 `datasets.load_dataset` 正常加载数据集， 这里会在执行前进行 barrier
                            3. 如果设置了列命映射 `columns` 则根据 `columns` 的键值对 `dataset = RowPreprocessor.safe_rename_columns(dataset, columns)` 对列名进行映射
                            4. `dataset = dataset_meta.preprocess_func(...)` 实际调用的是 `swift.llm.dataset.preprocessor.core.AutoPreprocessor.__call__(...)`
                                1. 如果设置了列命映射 `columns` 则根据 `columns` 的键值对 `dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)` 对列名进行映射
                                2. `preprocessor = self._get_preprocessor(dataset)`
                                    1. `features = dataset.features` 获取数据集所有 features
                                    2. 如果有 `['conversation', 'conversations', 'messages']` 字段，则返回 `MessagesPreprocessor(**self.kwargs)` (关键在这里！)
                                        1. 初始化主要是生成 `self.columns` 映射字典
                                    3. 如果有 `instruction` 和 `input` 字段，则返回 `AlpacaPreprocessor(**self.kwargs)`
                                    4. 否则返回 `ResponsePreprocessor(**self.kwargs)`
                                3. 执行实例化好的 `preprocessor`，实际执行的是 `RowPreprocessor.__call__(...)` 
                                    1. `dataset = sample_dataset(...)` 如果有采样参数就进行采样
                                    2. 如果不是 master 节点，设置 `load_from_cache_file = True` 让从节点从本地缓存加载
                                    2. `dataset = self._rename_columns(dataset)`
                                        1. 根据 `self.columns` 来对数据集字段进行名称映射
                                    2. `dataset = self.prepare_dataset(dataset)` 原封不动地返回 dataset
                                    3. `dataset = self._cast_pil_image(dataset)` 如果有图片字段，把对应字段的内容用 `Image` 读取为 `bytes` 类型
                                    4. `self._patch_arrow_writer()` 强制指定 `dataset.features` 为特定的结构和类型，比如图片字段的数类型为 `bytes` 等   
                                    5. `dataset_mapped = dataset.map(self.batched_preprocess, ...)`
                                        1. `map` 会先检查 cache 是否存在，如果存在，则会在生成 `dataset_num_proc` 个 pool 前直接读取数据，从而不走 `batched_preprocess`；否则生成 `dataset_num_proc` 个 pool 并行处理各自得到的部分数据
                                        2. 为了兼容流式处理，把 `__@` 前缀的特征去掉 `__@`
                                        3. 将批次列式数据（如 {'text': ['a', 'b']}）转换为行式数据（如 [{'text': 'a'}, {'text': 'b'}]）`rows = self.batched_to_rows(batched_row)`
                                        4. 逐条数据调用 `row = self.preprocess(row)` 实际调用的是 `MessagesPreprocessor.preprocess` 处理数据
                                            1. 单独取出 `messages`
                                            2. `self.repair_messages(messages)` 没啥用
                                            3. `self.to_std_messages(messages, system)` 标准化 `messages`
                                        5. 逐条数据进行各种校验，确保数据格式正确
                                        6. 将行式数据（如 [{'text': 'a'}, {'text': 'b'}]）转换为列式数据（如 {'text': ['a', 'b']}） `res = self.rows_to_batched(new_rows)`
                                        7. 把经过 `batched_preprocess` 处理、筛选后的结果返回
                            5. `dataset = RowPreprocessor.remove_useless_columns(dataset)` 
                                1. 取出 `dataset.features`
                                2. 只保留 `RowPreprocessor.standard_keys` 里的特征
                                3. 返回 `dataset`
                        2. 返回 `dataset`
                    2. `train_dataset, val_dataset = DatasetLoader.post_process(...)` 切分数据集
                        1. 计算验证集数量
                        2. `train_dataset, val_dataset = train_dataset.train_test_split(...)` 切分数据集
                        3. `train_dataset = sample_dataset(...)` 采样或洗牌数据集
                    3. `train_datasets = DatasetLoader._concat_datasets(train_datasets)` 训练集和验证集数据集拼接，本质上用的是 `datasets.concatenate_datasets`
                    4. `train_datasets = DatasetLoader.shuffle_dataset(...)` 训练集和验证集数据洗牌
                3. 如果传了验证集路径，就把用指定的验证集路径作为验证集
            2. Encode 数据：`train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)`
                1. 获取模板 template
                2. 保存验证集
                3. 实例化 `preprocessor = EncodePreprocessor(...)`
                4. `dataset = preprocessor(dataset, ...)` 使用 `__call__` 方法，实际用的是基类 `RowPreprocessor.__call__`
                    1. `dataset_mapped = dataset.map(self.batched_preprocess)`
                        1. `RowPreprocessor.batched_preprocess` 同上
                            1. ...
                            2. `row = self.preprocess(row)` 实际调用的是 `EncodePreprocessor.preprocess`
                                1. Encode: `encoded = self.template.encode(row, return_length=True)` 实际使用的是 `swift.llm.template.base.py` `Template.encode`
                                    1. `inputs, extra_kwargs = StdTemplateInputs.from_dict(inputs)`
                                        1. `tool_response` 规范化为 `tool`
                                        2. `tool_call` 的 content 如果不是 str，json.dumps 为 str
                                        3. `media_kwargs = StdTemplateInputs.remove_messages_media(messages)` 把 content 里的非文本信息（图片等 url）提取放到对应的字段里，文本里用 <image> 代替
                                        4. 返回 `StdTemplateInputs(...)` 的实例和保存了图片、音频、视频等其他信息的字典
                                    2. `self._preprocess_inputs(inputs)`
                                        1. `self._preprocess_function_call(inputs)`
                                            1. `agent_template = self.agent_template` 获取 agent_template
                                            2. `agent_template.template_meta = self.template_meta` 获取 template_meta
                                            3. 如果输入存在 tools 字段，则把 tools 字段的字符串 `_parse_json` 解析成 list of dict
                                            4. `inputs.tools[i] = agent_template.wrap_tool(tool)` 对每个工具都包装成 `{'type': 'function', 'function': tool}` 这种
                                            5. 最后对 `tool_call` 进行内容合并
                                        2. 如果是多模态：
                                            1. `self._replace_image_tags(inputs)`
                                            2. `self._replace_start_image_tags(inputs)`
                                    3. 根据任务不同使用不同的 encode 方法 `encoded = self._encode_truncated(inputs)`
                                        1. 如果是多模态：`self._add_default_tags(inputs)`
                                        2. Encode: `encoded = self._encode(inputs)`
                                            1. 如果 `template_backend` 是 swift `self._swift_encode(inputs) `; 否则用 `self._jinja_encode(inputs)`
                                                1. `template_meta = self.template_meta`
                                                2. `self._swift_prepare_inputs(inputs)`
                                                    1. `super()._swift_prepare_inputs(inputs)` 对连续的 user 或 assistant 或 tool 进行 content 合并
                                                3. `system = self._get_system(inputs)` 获取 system 内容
                                                    1. 获取 `input.system`
                                                    2. 如果 `tools` 有值：`system = self.agent_template._format_tools(tools, system or '', inputs.messages[0])` 把 tools 内容放入 system
                                                4. `self._get_std_messages(inputs.messages)` 如果是 pretrain，因为只有 assistant，标准化为 `messages.insert(0, {'role': 'user', 'content': ''})`; 如果 messages 是奇数，在最后追加 `{'role': 'assistant', 'content': None}`
                                                5. 如果 `template_meta.auto_add_bos`，则解析 `bos_token` 并添加到 `res_context_list`
                                                6. 如果没有 system: `prefix = template_meta.prefix`; 如果有 system: `prefix = template_meta.system_prefix`
                                                7. `self._concat_context_list(...)` 拼接一下
                                                8. ... 各种提取
                                                9. 返回 `res_context_list, loss_scale_list, answer_len`
                                            2. `res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)` 根据 `loss_scale_list` 合并 q, a，比如 `loss_scale_list = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]` 合并后为 `[0.0, 1.0, 0.0, 1.0]`
                                            3. `input_ids, labels, loss_scale = self._encode_context_list(res_context_list, loss_scale_list)`
                                                1. 对 q, a: `token_list = self._tokenize(context)`
                                                2. label 的 q 部分填充为 -100，a 保持原样
                                            4. `self._add_dynamic_eos(...)`
                                            5. `self._handle_megatron_cp(encoded)` 把 `input_ids` 填充到 `cp_size * 2` 的整数倍
                                            6. 返回 `encoded`
                                        3. 对于超过 `--max_length` 的按照 `self.truncation_strategy` 策略来截断
                                        4. 返回 `encoded`
                                    4. 加上 `length` 和 `template_inputs`，去掉无用的信息
                                2. 返回 encode 后的结果
            3. Concat 数据 `train_dataset = DatasetLoader._concat_datasets(train_datasets)`
            4. `datasets = [train_dataset, val_dataset]`
            5. Packing 数据 `dataset = packing_dataset_cls(...)` 实际调用的是 `PackingDataset` 的实例化
                1. 在主节点上 `self.packed_idx, self.packed_length = self.create_packed_idx()` 实际上使用 `binpacking` 包来计算哪些数据打包到一起
            6. 打印数据 `self._show_dataset(*datasets)` 调用 `self.template.print_inputs(inputs)` 打印 input_ids, input, labels_ids, labels
            7. 打印一下长度统计信息
            8. 返回 `datasets`
        2. `data_collator = self._get_data_collator()`
            1. `data_collator = self.template.data_collator` (`<bound method Template.data_collator of <swift.llm.template.template.utils.ThinkingTemplate object at 0x7fa4e6199350>>`)
            2. 如果 `tensor_model_parallel_size > 1` and `sequence_parallel`: `padding_to = args.tensor_model_parallel_size`
            3. 如果 `context_parallel_size > 1`: `padding_to = (padding_to or 1) * args.context_parallel_size`
            4. `data_collator` 的 `padding_to` 预填写好
        3. `self.trainer.train(train_dataset, val_dataset, data_collator)` 开始训练
            1. `datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)` 返回 `swift_datasets_provider` 这里已经包含了 `train_dataset, val_dataset`
            2. `self.patch_megatron_data_collator(data_collator)` 对 `training.build_pretraining_data_loader` 打补丁
            3. `self._get_iters(train_dataset, val_dataset)` 对 `training.initialize_megatron` 打补丁以兼容 `max_epochs` 的 `train_iters` 数
                1. `step_batch_size = args.micro_batch_size * data_parallel_size`
                2. `dataset_sample = len(train_dataset) // step_batch_size * step_batch_size`
                3. `args.train_iters = dataset_sample * args.max_epochs // args.global_batch_size`
            4. `pretrain(...)` 使用 Megatron 开始训练，其中 `args.megatron_model_meta.model_provider` 使用的是 `swift.megatron.model.gpt.model.py` 里的 `model_provider`
                1. `initialize_megatron(...)`
                    1. `validate_args(args, args_defaults)`
                        1. `decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size`
                        2. `args.data_parallel_size = args.world_size // total_model_size`
                        3. ...
                        4. 如果 `virtual_pipeline_model_parallel_size` 没设置，则禁用 `Overlap P2P communication`
                        5. 设置参数类型:
                            1. `args.main_grads_dtype`: torch.float32
                            2. `args.main_params_dtype`: torch.float32
                            3. `args.exp_avg_dtype`: torch.float32
                            4. `args.exp_avg_sq_dtype`: torch.float32
                            5. `args.params_dtype`: args.fp16 / args.bf16
                        6. ...
                        7. `_print_args("arguments", args)`
                    2. `set_global_variables(args)`
                        1. `set_args(args)` 把 `_GLOBAL_ARGS = args`
                        2. `init_num_microbatches_calculator(...)` -> `_configure_global_num_microbatches_calculator(...)`
                            1. `self.num_micro_batches = global_batch_size // (micro_batch_size * data_parallel_size)`
                        3. `_ = _build_tokenizer(args)`
                            1. `_GLOBAL_TOKENIZER = build_tokenizer(args)` -> `patch_megatron_tokenizer` -> `build_tokenizer(args)` 使用 HF 的 tokenizer
                    3. `setup_logging()`
                    4. `initialize_rerun_state_machine(...)`
                    5. `finish_mpu_init()`
                        1. `_initialize_distributed(...)`
                            1. 如果 `torch.distributed.is_initialized()` 没有初始化，则先初始化
                            2. `mpu.initialize_model_parallel(...)` 设置模型并行，数据并行等各种进程组，每个 rank 对应进程都有自己全局变量
                                1. `decoder_rank_generator = RankGenerator(...)` order='tp-cp-ep-dp-pp'
                                2. `expert_tensor_model_pipeline_parallel_size = (expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size)`
                                4. `expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size`
                        2. `_set_random_seed(...)` 设置随机种子
                    6. `_init_autoresume()` 自动恢复
                    7. `_compile_dependencies()`
                2. `set_jit_fusion_options`
                3. `model, optimizer, opt_param_scheduler = setup_model_and_optimizer(...)`：初始化时打的补丁替换，用 `swift.megatron.trainers.base.py` 里的 `setup_model_and_optimizer` 替换 Megatron 原有的 `setup_model_and_optimizer`
                    1. `with self._patch_load_state_dict()` 
                        1. `checkpointing._load_base_checkpoint = _load_base_checkpoint` 把 Megatron 的 `checkpointing._load_base_checkpoint` 替换，实现对模型参数 (`sharded_state_dict['model']`) 打高效 cat 的 `_patch_merge_fn` 补丁
                        2. lora 时做一些 key 映射
                        3. 离开上下文后进行还原
                    2. `model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(new_model_provider_func, ...)` 使用 Megatron 原本的 `megatron.training.training.py` 里的 `setup_model_and_optimizer`
                        1. `model = get_model(model_provider_func, model_type)`
                            1. `model = build_model()`
                                1. `model = model_provider_func(pre_process=pre_process, post_process=post_process)` 实际用的是打补丁的 `swift.megatron.trainers.base.py` 里的 `setup_model_and_optimizer` 里的 `new_model_provider_func`
                                    1. `self.unwrapped_model = model_provider_func(*args, **kwargs)` 正常调用原本的 `swift.megatron.model.gpt.model.py` 的 `model_provider_func` 得到 `self.unwrapped_model` 基础模型
                                        1. `config = core_transformer_config_from_args(args)` 得到 `TransformerConfig` 配置实例
                                        2. 如果是 MoE: `transformer_layer_spec = get_gpt_decoder_block_spec(...)`
                                            1. `layer_norm_impl = TENorm`
                                            2. `dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)`
                                                1. `backend = TESpecProvider()`
                                                2. `mlp = get_mlp_module_spec_for_backend(...)`
                                                    1. `linear_fc2 = backend.row_parallel_linear()` 返回 `TERowParallelLinear`
                                                    2. `linear_fc1 = backend.column_parallel_layer_norm_linear()` 返回 `TELayerNormColumnParallelLinear`
                                                    3. 返回 MLP 的 `ModuleSpec(module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2))`
                                                3. `qk_norm = backend.layer_norm(for_qk=True)` 返回 `TENorm`
                                                4. 有了 Dense mlp，加上通用的 self-attention 层，返回 TransformerLayer 的 `ModuleSpec(...)`
                                            3. `moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)`
                                                1. `backend = TESpecProvider()`
                                                2. `mlp = get_mlp_module_spec_for_backend(...)`
                                                    1. `linear_fc2 = backend.row_parallel_linear()` 返回 `TERowParallelLinear`
                                                    2. 返回 `get_moe_module_spec_for_backend(...)`
                                                        1. 共享专家：
                                                            1. `linear_fc1 = backend.column_parallel_linear()` 返回 `TEColumnParallelLinear`
                                                            2. `linear_fc2 = backend.row_parallel_linear()` 返回 `TERowParallelLinear`
                                                            3. `mlp = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)`
                                                            4. `shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)`
                                                        2. 普通专家：
                                                            1. `expert_module, expert_submodule = backend.grouped_mlp_modules(...)` 返回 `TEGroupedMLP, MLPSubmodules(linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear)`
                                                            2. `experts = ModuleSpec(module=expert_module, submodules=expert_submodule)`
                                                        3. 组成 MoE 层：`moe_module_spec = ModuleSpec(module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts))`
                                                3. `qk_norm = backend.layer_norm(for_qk=True)` 返回 `TENorm`
                                                4. 有了 MoE mlp，加上通用的 self-attention 层，返回 TransformerLayer 的 `ModuleSpec(...)`
                                            4. 遍历所有层 `config.num_layers` 根据 `moe_layer_pattern`(`config.moe_layer_freq`: 0 是 Dense; 1 是 MoE) 向 `layer_specs` 添加对应类型的模型模块
                                            5. `num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)` 即 `num_layers // config.pipeline_model_parallel_size`
                                            6. `offset = get_transformer_layer_offset(config, vp_stage=vp_stage)` 计算 offset
                                            7. 根据 offset 从 `layer_specs` 索引出对应的层
                                            8. 返回 PP 下当前卡上对应的 block `block_spec = TransformerBlockSubmodules(...)`
                                        3. 如果使用 `use_shared_expert_gate` 追加相应的参数
                                        4. `model = GPTModel(...)` 使用的是 `swift.megatron.model.gpt_model.py` 里的 `GPTModel`，model 绑定了 `TransformerConfig`
                                            1. 调用父类 `McoreGPTModel` 的初始化方法，实际上就是 `Megatron-LM.megatron.core.models.gpt.get_model.py` 里的 `GPTModel`
                                                1. `self.embedding = LanguageModelEmbedding(...)`
                                                2. `self.rotary_pos_emb = RotaryEmbedding(...)`
                                                3. `self.decoder = TransformerBlock(...)`
                                                    1. `get_cpu_offload_context(...)`
                                                    2. `model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()` 获取通信组
                                                    3. `self._build_layers()` 构建网络
                                                        1. 遍历 `self.submodules.layer_specs` 里 pp 切分后当前 rank 上的所有层，每一层应用 `build_layer(layer_spec, i + 1)`
                                                            1. 计算全局 `global_layer_number`
                                                            2. `module = build_module(...)`
                                                                1. `module = spec_or_module.module` 获取构建模型实例的类名
                                                                2. `module(...)`
                                                                    1. `self.input_layernorm = build_module(...)` 递归调用 `build_module` 构建模型各层（即，把每个 module 的初始化方法都构造好，有些 module 的初始化方法里会调用 `build_module`，因此就形成了递归调用）
                                                        2. 把每一个 TransformerLayer 用 `torch.nn.ModuleList` 组装起来
                                                4. `self.setup_embeddings_and_output_layer()` 设置 Embedding 权重的属性，标记成 `is_embedding_or_output_parameter = True`
                                            2. `new_inv_freq` 放到指定设备上
                                            3. 返回 model (`self.unwrapped_model`)，此时显存已经加载了模板模型
                                    2. `self.peft_model = prepare_mcore_model(self.unwrapped_model)` 这个是 `new_model_provider_func` 的关键，打上 LoRA 相关的补丁
                                        1. 如果是 `full`，根据 `freeze_parameters` 相关参数冻结网络层，根据 `trainable_parameters` 相关参数训练指定网络层
                                        2. **如果是 `lora`，`model = prepare_adapter(model)`**
                                            1. `set_linear_is_expert(model)` 把专家层标记一下
                                            2. `target_modules = get_target_modules(args, model)` 获取 LoRA 目标 module e.g. `['o_proj', 'v_proj', 'gate_proj', 'q_proj', 'down_proj', 'k_proj', 'up_proj']`
                                            3. `modules_to_save = get_modules_to_save(args, model)` 获取要保存的模块
                                            4. `lora_config = LoraConfig(...)` 生成 LoRA 配置项
                                            5. `model = Swift.prepare_model(model, lora_config)`
                                                1. 如果 `config` 类型是 `SwiftConfig` 或 `dict`: `SwiftModel(model, config, **kwargs)`
                                                2. **如果 `config` 类型是其他**，如 `LoraConfig`: 用 peft 的 `get_peft_model(model, config, **kwargs)` 这里传入的 model 是经过 pp 切分后的部分模型 -> `peft.peft_model.PeftModelForCausalLM`
                                                    1. `self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)` cls 为 `peft.tuners.lora.model.LoraModel` 打的初始化补丁
                                                        1. 用原本的 `LoraModel` 初始化方法初始化 -> `peft.tuners.tuners_utils.py` BaseTuner 的初始化
                                                            1. `self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)`
                                                                1. 从 `peft_config` 里看当前层名是否属于 `peft_config.target_modules`
                                                                2. 如果属于则调用 `self._create_and_replace(...)` -> 打的补丁 `self._create_and_replace_origin` -> 原本的 `_create_and_replace`
                                                                    1. `new_module = self._create_new_module(...)` -> `dispatch_megatron(...)` -> `new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs)` 创建 Megatron 格式的新 LoRA 层并返回新的层
                                                                    2. `self._replace_module(...)` 用新的层替换原本的层，这是 LoRA 的核心
                                                            2. `self.inject_adapter` 后的 `self.model` 就把原本的线性目标层替换为 LoRA 层
                                                        2. 如果设置了 LoRA 的 dtype 类型，对 LoRA 层进行类型转换
                                                        3. `self.base_model._cast_adapter_dtype` 把 LoRA 层的参数转换为 float32
                                            6. `return model` 此时就是替换为 LoRA 后的模型
                                    3. 返回返回实例化好并且打好 LoRA 补丁的模型
                            2. 遍历 model 里的每一个参数，调用 `tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)` 设置参数张量并行属性 (`_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS`)，保持一致的属性定义
                            3. 把模型每个参数转移到 cuda
                            4. 对于 fp16 或 bf16:
                                1. `config = get_model_config(model[0])` 获取设置
                                2. 调用 `Float16Module`
                                    1. `self.add_module('module', module.bfloat16())` 把 module 加入 `self._modules['module']` 里
                                    2. 返回 Float16Module 包装后的模型
                            5. 把跟分布式相关的参数提取到 kwargs
                            6. `ddp_config = DistributedDataParallelConfig(**kwargs)` 根据 kwargs DDP 生成分布式配置实例
                            7. 返回被 DDP 包裹后的 model
                        2. `unwrapped_model = unwrap_model(model)` 把 model 层级解开，解到 GPTModel 这个 module
                        3. 取出优化器配置 `OptimizerConfig`
                        4. 获取优化器 `optimizer = get_megatron_optimizer(...)`
                            1. `param_groups, buffers = _get_param_groups_and_buffers(...)`
                                1. `param_groups = _get_param_groups(...)` 注意，通过 param 的 `allreduce` 属性区分 dense 还是 expert，`allreduce=True` 表示 dense，否则为 expert；bias 和 norm 参数不做 regularize，返回包含区分 dense 和 expert 的 `param_groups`
                            2. `_get_megatron_optimizer_based_on_param_groups(...)`
                                1. `optimizer = Adam(**kwargs)` 实例化优化器
                                2. `optimizer = DistributedOptimizer(...)` 分布式
                            3. MoE 再来一遍
                            4. `ChainedOptimizer(optimizers)`
                        5. 获取优化器调度器 `opt_param_scheduler = get_optimizer_param_scheduler(optimizer)`
                        6. 如果设置了 `--moe-use-upcycling` (升维转换：从一个已经训练好的、标准的密集型模型（Dense Model）来初始化一个更大、更复杂的混合专家模型（Mixture-of-Experts, MoE Model）。这是一种常见的技术，用于在开始训练一个庞大的MoE模型之前，给它一个更好的权重初始值，而不是从零开始随机初始化)
                            1. `args.ffn_hidden_size` 乘以一个 `moe_upcycling_granularity`（升维粒度）因子，暂时让程序认为它要构建的是一个标准的、没有专家的密集模型，并且这个模型的FFN层比原始的要“胖”得多。这个“胖”的FFN层之后将被用来拆分并形成多个专家
                            2. 调用 `dense_model_for_upcycling = get_model(model_provider_func, model_type)` 函数来创建一个临时的、更“胖”的密集模型实例
                            3. 将 args 恢复成原始的MoE模型配置
                            4. `upcycling_utils.load_and_upcycle_model` 核心升维转换
                                1. 将一个预训练好的密集模型的权重加载到刚刚创建的 `dense_model_for_upcycling` 中，基础的 Dense 模型由 `--load` 参数指定
                                2. 转换和复制：将这个临时密集模型的“胖”FFN层的权重进行拆分、变换或复制，用来初始化当前MoE模型 (unwrapped_model) 中的多个专家（experts）
                            5. 代码将迭代步数 `args.iteration` 强制设为 1，然后调用 `save_checkpoint` 将这个全新的 MoE 模型保存为一个检查点，这个检查点就是接下来训练 MoE 模型的起点
                            6. `del dense_model_for_upcycling` 删除临时的密集模型以释放显存
                            7. `optimizer.reload_model_params()` 如果使用了fp16或bf16混合精度训练，由于模型参数被直接修改了，需要调用这个方法来确保优化器内部的状态和新的模型参数是同步的
                        7. 如果配置了基础模型 `args.load` 或 `args.pretrained_checkpoint`：`load_checkpoint(model, optimizer, opt_param_scheduler, ...)` 开始加载模型权重
                            1. `state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(...)` 此时激活了补丁里的 `with self._patch_load_state_dict()`，会调用补丁里的 `_load_base_checkpoint`
                                1. 打一些合并的补丁，再调用 `origin__load_base_checkpoint`，即 `megatron.training.checkpointing.py` 里的 `_load_base_checkpoint`
                                    1. 读取 `latest_checkpointed_iteration.txt` 里的迭代步数
                                    2. 根据 `ckpt_format` 的不同类型（`torch_dist`, `torch`, `torch_dcp`）使用不同的加载方法，比如 `_load_global_dist_base_checkpoint`
                                        1. 获取 `checkpoint_name` ('/models/ZhipuAI/GLM-4.5-Air-mcore/iter_0000001')
                                        2. `state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)` -> `common_strategy.load_common(checkpoint_dir)`
                                            1. `torch.load(load_path, map_location='cpu', weights_only=False)` 加载 `/models/ZhipuAI/GLM-4.5-Air-mcore/iter_0000001/common.pt` 为 `state_dict`
                                        3. 返回 `state_dict, checkpoint_name, release, CheckpointType.GLOBAL`
                                2. 返回 `state_dict, checkpoint_name, release, CheckpointType.GLOBAL`
                            2. `sharded_sd_metadata = dist_checkpointing.load_content_metadata(...)`
                            3. `load_kwargs['sharded_state_dict'] = generate_state_dict(...)` 这个函数并不会加载数据，而是根据当前模型的结构，生成一个空的 state\dict 结构（模板）。这个模板告诉后续的加载函数，当前进程需要加载哪些张量以及它们的形状和分布情况
                                1. `model_sd = model[i].sharded_state_dict(**(model_sd_kwargs or {}))`
                                2. 放入 `state_dict` 里
                                3. 返回 `state_dict` -> `load_kwargs['sharded_state_dict'] = state_dict`
                            4. 实际加载: `state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(...)` 准备好模板后，再次调用，注意，此时 `rank0=False`, `sharded_state_dict` 不为空了，所有 rank 都参与，并将模板作为 `sharded_state_dict` 参数传入，这个函数会根据模板，从磁盘读取相应的分片数据，并将它们填充到模板中，当 `_load_base_checkpoint` 执行完毕后，`state_dict` 变量中就包含了从磁盘读取到的所有状态数据
                                1. 把 `sharded_state_dict['model']` 里的每一层模型层名称替换，把 `.base_layer` 替换为空，同时记录到 `state_dict_model` 映射里
                                2. `self._patch_merge_fn(state_dict_model)`
                                    1. 对于 `ShardedTensorFactory` 的张量，替换 `v.merge_fn = sh_ten_merge_fn`
                                3. `res = origin__load_base_checkpoint(*_args, **kwargs)`，即 `megatron.training.checkpointing.py` 里的 `_load_base_checkpoint` -> `_load_global_dist_base_checkpoint` 此时 `rank0=False`
                                    1. `state_dict = dist_checkpointing.load(...)`
                                        1. `common_state_dict = common_strategy.load_common(checkpoint_dir)` 加载静态字典
                                        2. `sharded_state_dict, _ = extract_sharded_base(...)` 提取出 `ShardedBase` 的模型层
                                        3. `local_metadata, global_metadata = determine_global_metadata(sharded_state_dict)`
                                            1. `local_metadata = [ten.without_data() for ten in nested_values(sharded_state_dict)]` 获取本卡的元数据
                                            2. `torch.distributed.all_gather_object(global_metadata, local_metadata)` all-gather 拿到所有卡的元数据
                                        4. `loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)` 加载属于该卡的模型权重 -> 内部使用 `checkpoint.load_state_dict(...)` 加载模型，有个 barrier
                                        5. `loaded_state_dict = apply_factory_merges(...)` 使用 `sh_ten_merge_fn` 补丁高效执行 SwiGLU 的 cat 操作
                                        6. 返回 `common_state_dict`
                                4. 加载后的模型 state_dict 再还原回原来的层名
                            5. 接下来是一系列检查操作：
                                1. 设置迭代信息: 从 `state_dict` 中恢复 `iteration`, `num_floating_point_operations_so_far` 等训练状态，如果是在 `finetune` 微调模式下，`iteration` 会被重置为 0
                                2. 检查参数一致性: 检查当前运行的参数和检查点中保存的参数是否兼容
                                3. 恢复模型权重: 除非 `skip_load_to_model_and_opt` 为 True，否则会调用 `model.load_state_dict()` 将 `state_dict` 中的权重加载到 DDP 模型中。这里有一个 try-except 的回退机制，如果严格加载失败，会尝试非严格加载，以兼容某些库（如 TransformerEngine）的向后不兼容更改
                                4. 修复 QKV 矩阵顺序: 调用 `fix_query_key_value_ordering` 来处理不同版本 Megatron 中 Attention 模块权重顺序可能不一致的问题
                                5. 恢复优化器和学习率调度器状态: 如果不是微调模式，且没有指定 `--no-load-optim`，则会加载优化器和调度器的状态
                                6. 恢复 RNG 状态: 如果不是微调模式，且没有指定 `--no-load-rng`，则会恢复 Python random, numpy, torch, torch.cuda 的随机数生成器状态。这是保证从断点继续训练时数据加载、dropout 等随机过程可复现的关键
                        8. 返回 `model, optimizer, opt_param_scheduler`
                    3. 如果 `train_type` 不是 `full` 并且有 `args.modules_to_save`，拷贝原始模型 `copy_original_module_weight(self.unwrapped_model)`
                    4. 如果需要 `initialize_embedding`，把原始模型的 Embedding 层权重初始化 `self._initialize_embedding(self.unwrapped_model)`
                    5. 返回 `model, optimizer, opt_param_scheduler` 
                4. `build_train_valid_test_data_iterators(train_valid_test_dataset_provider)` 构建 data_iterator，这里的 `train_valid_test_dataset_provider` 就是 pretrain 的 `datasets_provider`
                    1. `train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)`
                        1. `train_ds, valid_ds, test_ds = build_train_valid_test_datasets(build_train_valid_test_datasets_provider)` -> `build_train_valid_test_datasets_provider(train_valid_test_num_samples)` -> 实际调用的是 `swift.megatron.trainers.utils.py` 里的 `get_swift_datasets_provider` 里的 `swift_datasets_provider` 基本上就是把 `train` 里 `datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)` 包装的数据集，再取出来
                        2. `train_dataloader = build_pretraining_data_loader(train_ds, args.consumed_train_samples)` -> 实际调用的是 `swift.megatron.trainers.base.py` 里打的补丁 `patch_megatron_data_collator` 里的 `build_pretraining_data_loader`
                            1. `res = origin_build_pretraining_data_loader(*_args, **kwargs)`
                                1. `batch_sampler = MegatronPretrainingRandomSampler(...)`
                                2. 返回 `res = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, ...)`
                            2. `res.collate_fn = data_collator` 绑定 `Template.data_collator`
                        3. `valid_dataloader`, `test_dataloader` 同理
                        4. `torch.distributed.broadcast(flags, 0)` 把是否做相关的 dataloader 广播出去
                        5. `return train_dataloader, valid_dataloader, test_dataloader`
                    2. `train_data_iterator = _get_iterator(dl_type, train_dataloader)`
                    3. `valid_data_iterator`, `test_data_iterator` 同理
                    4. `return train_data_iterator, valid_data_iterator, test_data_iterator`
                5. `iteration, num_floating_point_operations_so_far = train(...)` 开始训练
                    1. 取出 model 里的 model_module `model_module.train()`
                    2. `train_step(forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config)`
                        1. `losses_reduced = forward_backward_func(...)` 调用的是 `megatron.core.pipeline_parallel.schedules.py` 里的 `forward_backward_pipelining_without_interleaving` -> `output_tensor, num_tokens = forward_step(...)` -> `output_tensor, loss_func = forward_step_func(data_iterator, model)` -> `MegatronTrainer` 的 `forward_step` `output_tensor = model(**data)` -> `_BaseDataParallel` 的 `forward` `self.module(*inputs, **kwargs)` -> `Float16Module` 的 `forward` `outputs = self.module(*inputs, **kwargs)` -> `swift.megatron.model.gpt_model.py` `GPTModel` 的 `forward` 
                            1. 如果 `self.pre_process`: `decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)` 这里的 `input_ids.shape` 为 `torch.Size([1, 2048])`, `position_ids.shape` 为 `torch.Size([1, 2048])`， 都是 `max_length / context_parallel_size` 的大小 
                            2. `rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(...)`
                            3. `rotary_pos_emb = self.rotary_pos_emb(...)`
                            4. `hidden_states = self.decoder(...)` 调用的是 `TransformerBlock` 的 `forward`
                                1. `hidden_states = self._checkpointed_forward(...)` -> ... -> `swift.megatron.tuners.lora.py` 里的 `LoraParallelLinear` 的 `forward`
                                    1. `(result, x), bias = self.base_layer(x, *args, **kwargs)` 先计算 base_layer 的结果
                                    2. 遍历 `self.active_adapters` 里的 adapter
                                        1. 取出 `lora_A`, `lora_B`, `dropout`, `scaling`, `dtype`
                                        2. `lora_result = lora_A(dropout(x))`
                                        3. `lora_result = lora_B(lora_result)`
                                        4. `lora_result = lora_result * scaling`
                                        5. `result = result + lora_result`
                                    3. `result = result.to(previous_dtype)`
                                    4. `return result, bias`
                            5. 如果共享 embedding `self.share_embeddings_and_output_weights`: `output_weight = self.shared_embedding_or_output_weight()`
                            6. 如果不共享 embedding `logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)`
                            7. `loss = self.compute_language_model_loss(labels, logits)`
                            8. `return loss`





                    
                        











