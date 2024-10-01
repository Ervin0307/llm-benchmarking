import re
from enum import Enum


class ProfileMethod(str, Enum):
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"


class ProfileLayerBase:
    @classmethod
    def get_available_profile_names(cls):
        return [key.name.lower() for key in cls]

    @classmethod
    def get_profile_names_by_module(cls, module_name: str):
        """
        Get all operations related to a specific module (e.g., self_attn, mlp).
        """
        return [
            attr.name.lower()
            for attr in cls
            if re.search(
                "\\b"
                + "|".join(pattern.rsplit(".", 1)[0] for pattern in attr.value)
                + "$",
                module_name,
            )
        ]

    @classmethod
    def get_profile_name_by_operation(cls, full_layer_path: str):
        full_layer_path = full_layer_path.lower().rsplit(":", 1)[0]
        if not full_layer_path:
            raise ValueError(
                "The full_layer_path should be in the format 'module_name.operation_name' or 'operation_name'."
                " For example, 'llamaforcausallm.model.layers.25.self_attn.rotary_emb' "
                "or 'attn_input_reshape'."
            )

        for attr in cls:
            if re.search("\\b" + "|".join(attr.value) + "$", full_layer_path):
                return attr.name.lower()


class VllmProfileLayer(ProfileLayerBase, Enum):
    # Attention Ops
    ATTN_INPUT_RESHAPE = ("attn_backend.attn_input_reshape",)
    ATTN_KV_CACHE_SAVE = ("attn_backend.attn_kv_cache_save",)
    ATTN_PREFILL = ("attn_backend.attn_prefill",)
    ATTN_DECODE = ("attn_backend.attn_decode",)
    ATTN_OUTPUT_RESHAPE = ("attn_backend.attn_output_reshape",)

    # Attention Layers
    EMB = ("model.embed_tokens",)
    INPUT_LAYERNORM = ("layers.\d.input_layernorm",)
    ATTN_PRE_PROJ = ("self_attn.qkv_proj",)
    ATTN_ROPE = ("self_attn.rotary_emb",)
    ATTN_POST_PROJ = ("self_attn.o_proj",)
    POST_ATTENTION_LAYERNORM = ("layers.\d.post_attention_layernorm",)

    # MLP Layers
    MLP_UP_PROJ = ("mlp.gate_up_proj", "mlp.up_proj")
    MLP_ACT = ("mlp.act_fn",)
    MLP_DOWN_PROJ = ("mlp.down_proj",)
    ADD = ("decoder.add",)
