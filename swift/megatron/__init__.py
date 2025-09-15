# Copyright (c) Alibaba, Inc. and its affiliates.

try:
    from .init import init_megatron_env
    init_megatron_env()
except Exception:
    # allows lint pass.
    raise

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # NOTE: 给 IDE 和类型检查工具看
    from .train import megatron_sft_main, megatron_pt_main, megatron_rlhf_main
    from .utils import convert_hf2mcore, convert_mcore2hf, prepare_mcore_model, adapter_state_dict_context
    from .argument import MegatronTrainArguments, MegatronRLHFArguments
    from .model import MegatronModelType, MegatronModelMeta, get_megatron_model_meta, register_megatron_model
    from .trainers import MegatronTrainer, MegatronDPOTrainer
    from .tuners import LoraParallelLinear
else:
    # NOTE: 程序真正运行时执行的逻辑
    # 特点：
    # 1. 不立即导入
    # 2. 定义模块结构 (_import_structure)
    _import_structure = {
        'train': ['megatron_sft_main', 'megatron_pt_main', 'megatron_rlhf_main'],
        'utils': ['convert_hf2mcore', 'convert_mcore2hf', 'prepare_mcore_model', 'adapter_state_dict_context'],
        'argument': ['MegatronTrainArguments', 'MegatronRLHFArguments'],
        'model': ['MegatronModelType', 'MegatronModelMeta', 'get_megatron_model_meta', 'register_megatron_model'],
        'trainers': ['MegatronTrainer', 'MegatronDPOTrainer'],
        'tuners': ['LoraParallelLinear'],
    }

    import sys

    # NOTE: 惰性加载：通过用一个轻量的代理对象 (_LazyModule) 替换自身，实现了惰性加载，只在真正需要某个功能时才去加载对应的代码
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
