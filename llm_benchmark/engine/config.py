

class EngineConfig:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1, # set 0 if not supported
        pipeline_parallel_size: int = 1, # set 0 if not supported
        kv_cache_dtype: str = "float16",
        distributed_backend: str = "mp", # mp, ray
        gpu_memory_utilization: float = 0.8,
        swap_space: int = 4, # in GB
        cpu_kv_cache_size: int = 40, # in GB
        block_size: int = 16,
        max_num_batched_tokens: int = 1024,
        max_num_seqs: int = 1024,
        enable_prefix_cache: bool = True,
        enable_chunked_prefill: bool = False,
        disable_sliding_window: bool = False,
        quantization: str = None, # fp16, int8, int4, etc.
        rope_scaling: dict = None, # {type: linear, factor: 1.0},
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 4096,
        num_scheduler_steps: int = 1,
        scheduler_delay_factor: float = 0.0,
        attention_backend: str = "flash_attn", # flash_attn, flash_infer, torch_sdpa
    ):

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.kv_cache_dtype = kv_cache_dtype
        self.distributed_backend = distributed_backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.cpu_kv_cache_size = cpu_kv_cache_size
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_cache = enable_prefix_cache
        self.enable_chunked_prefill = enable_chunked_prefill
        self.disable_sliding_window = disable_sliding_window
        self.quantization = quantization
        self.rope_scaling = rope_scaling
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.num_scheduler_steps = num_scheduler_steps
        self.scheduler_delay_factor = scheduler_delay_factor
        self.attention_backend = attention_backend

    def get_config(self):
        return self.__dict__
        
        
def get_config_from_vllm(config: dict, envs: dict):
    return EngineConfig(
        model_name=config.get("model"),
        tensor_parallel_size=config.get("tensor_parallel_size"),
        pipeline_parallel_size=config.get("pipeline_parallel_size"),
        kv_cache_dtype=config.get("kv_cache_dtype"),
        distributed_backend=config.get("distributed_executor_backend"),
        gpu_memory_utilization=config.get("gpu_memory_utilization"),
        swap_space=config.get("swap_space"),
        cpu_kv_cache_size=envs.get("VLLM_CPU_KVCACHE_SPACE"),
        block_size=config.get("block_size"),
        max_num_batched_tokens=config.get("max_num_batched_tokens"),
        max_num_seqs=config.get("max_num_seqs"),
        enable_prefix_cache=config.get("enable_prefix_cache"),
        enable_chunked_prefill=config.get("enable_chunked_prefill"),
        disable_sliding_window=config.get("disable_sliding_window"),
        quantization=config.get("quantization"),
        rope_scaling=config.get("rope_scaling"),
        enforce_eager=config.get("enforce_eager"),
        max_seq_len_to_capture=config.get("max_seq_len_to_capture"),
        num_scheduler_steps=config.get("num_scheduler_steps"),
        scheduler_delay_factor=config.get("scheduler_delay_factor"),
        attention_backend=config.get("VLLM_ATTENTION_BACKEND"),
    )

def get_config_from_sglang(config: dict, envs: dict):
    return EngineConfig(
        model_name=config.get("model_name"),
        tensor_parallel_size=config.get("tensor_parallel_size"),
        pipeline_parallel_size=config.get("pipeline_parallel_size"),
        kv_cache_dtype=config.get("kv_cache_dtype"),
        distributed_backend=config.get("distributed_backend"),
        gpu_memory_utilization=config.get("gpu_memory_utilization"),
        swap_space=config.get("swap_space"),
        cpu_kv_cache_size=config.get("cpu_kv_cache_size"),
        block_size=config.get("block_size"),
        max_num_batched_tokens=config.get("max_num_batched_tokens"),
        max_num_seqs=config.get("max_num_seqs"),
        enable_prefix_cache=config.get("enable_prefix_cache"),
        enable_chunked_prefill=config.get("enable_chunked_prefill"),
        disable_sliding_window=config.get("disable_sliding_window"),
        quantization=config.get("quantization"),
        rope_scaling=config.get("rope_scaling"),
        enforce_eager=config.get("enforce_eager"),
        max_seq_len_to_capture=config.get("max_seq_len_to_capture"),
        num_scheduler_steps=config.get("num_scheduler_steps"),
        scheduler_delay_factor=config.get("scheduler_delay_factor"),
        attention_backend=envs.get("VLLM_ATTENTION_BACKEND"),
    )



def get_engine_config(engine: str, config: dict, envs: dict):
    if engine == "vllm":
        return get_config_from_vllm(config, envs)
    elif engine == "sglang":
        return get_config_from_sglang(config, envs)
    else:
        raise ValueError(f"Engine {engine} not supported")
