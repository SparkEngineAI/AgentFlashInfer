from transformers import PretrainedConfig
from vllm.config import ModelConfig, SpeculativeConfig
import vllm.envs as envs
from pangu_infer.pangu.adaptor_manager.adaptor_manager import register
from typing import Any
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_attr_by_names(src_config, attrs, default_value):
    for attr in attrs:
        value = getattr(src_config, attr, 0)
        if value > 0:
            return value
    return default_value


# @register("vllm.config.ModelConfig.is_deepseek_mla.fget")
@property
def is_deepseek_mla(self) -> bool:
    kv_lora_dim_names = ["attention_kv_lora_dim", "kv_lora_rank"]
    kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, None)
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_mtp",
            "pangu_ultra_moe",
    ):
        return kv_lora_dim is not None
    elif self.hf_text_config.model_type == "eagle":
        # if the model is an EAGLE module, check for the
        # underlying architecture
        return (
                self.hf_text_config.model.model_type in ("deepseek_v2", "deepseek_v3", "pangu_ultra_moe")
                and kv_lora_dim is not None
        )
    return False


# this is a new function for pangu 72Bv2: v head dim is different from q and k
@register("vllm.config.ModelConfig.is_different_kvdim")
def is_different_kvdim(self) -> int:
    v_dim_names = ['v_channels', 'v_head_dim']
    v_head_dim = get_attr_by_names(self.hf_text_config, v_dim_names, None)
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in ('PanguProMoE'):
        return v_head_dim is not None
    return False


@register("vllm.config.ModelConfig.get_head_size")
def get_head_size(self) -> int:
    if self.is_deepseek_mla:
        qk_rope_dim_names = ["attention_qk_rope_dim", "qk_rope_head_dim"]
        kv_lora_dim_names = ["attention_kv_lora_dim", "kv_lora_rank"]
        qk_rope_head_dim = get_attr_by_names(self.hf_text_config, qk_rope_dim_names, 0)
        kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, 0)

        if self.use_mla:
            return kv_lora_dim + qk_rope_head_dim
        else:
            qk_nope_dim_names = ["attention_qk_dim", "qk_nope_head_dim"]
            qk_nope_head_dim = getattr(self.hf_text_config, qk_nope_dim_names, 0)
            if qk_rope_dim and qk_nope_head_dim:
                return qk_rope_dim + qk_nope_head_dim

    if hasattr(self.hf_text_config, "model_type") and (self.hf_text_config.model_type == "zamba2"):
        return self.hf_text_config.attention_head_dim

    if self.is_attention_free:
        return 0

    # NOTE: Some configs may set head_dim=None in the config
    if getattr(self.hf_text_config, "head_dim", None) is not None:
        return self.hf_text_config.head_dim

    # FIXME(woosuk): This may not be true for all models.
    return self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads


@register("vllm.config.ModelConfig._verify_with_expert_parallelism")
def _verify_with_expert_parallelism(self) -> None:
    num_expert_names = [
        "moe_num_experts",  # Dbrx
        "num_experts",  # Jamba
        "n_routed_experts",  # DeepSeek
        "num_local_experts",  # Mixtral
    ]

    # patch code
    extra_num_expert_names = [
        "num_routed_experts",  # Pangu
    ]
    num_expert_names.extend(extra_num_expert_names)

    num_experts = 0
    for name in num_expert_names:
        num_experts = getattr(self.hf_text_config, name, 0)
        if num_experts > 0:
            break
    if num_experts < 1:
        raise ValueError("Number of experts in the model must be greater than 0 " "when expert parallelism is enabled.")


@register("vllm.config.SpeculativeConfig.hf_config_override")
@staticmethod
def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    if hf_config.model_type == "deepseek_v3":
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]})

    if hf_config.architectures[0] == "MiMoForCausalLM":
        hf_config.model_type = "mimo_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"],
            }
        )
        return hf_config

    # patch code
    if hf_config.model_type == "pangu_ultra_moe":
        hf_config.model_type = "deepseek_mtp"
        n_predict = getattr(hf_config, "num_mtp_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]})

    if hf_config.model_type == "PanguProMoE":
        hf_config.model_type = "deepseek_mtp"
        n_predict = getattr(hf_config, "num_mtp_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["PanguProMoEMTP"]})

    if hf_config.model_type == "openpangu_vl":
        hf_config.model_type = "deepseek_mtp"
        n_predict = getattr(hf_config, "num_mtp_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["PanguProMoEMTP"]})

    return hf_config


@register("vllm.config.SpeculativeConfig.__post_init__")
def __post_init__(self):

    # Note: "method" is a new parameter that helps to extend the
    # configuration of non-model-based proposers, and the "model" parameter
    # will be used to set the draft model, eagle head, or additional weight
    # when needed. If users do not specify "method", the speculative method
    # will be detected automatically if possible. If the speculative method
    # can not be detected, it will be considered as the "draft_model" by
    # default.

    if self.model is None and self.num_speculative_tokens is not None:
        # TODO(Shangming): Refactor mtp configuration logic when supporting
        # mtp acceleration for more models besides deepseek_v3
        # patch code
        if self.target_model_config and (
            self.target_model_config.hf_text_config.model_type == "deepseek_v3"
            or self.target_model_config.hf_text_config.model_type == "mimo"
        ):
            # use the draft model from the same model:
            self.model = self.target_model_config.model
        elif (self.target_model_config 
            and self.target_model_config.hf_text_config.model_type == "pangu_ultra_moe"
        ):
            self.model = self.target_model_config.model
        elif (self.target_model_config 
            and self.target_model_config.hf_text_config.model_type == "PanguProMoE"
        ):
            self.model = self.target_model_config.model
        elif (self.target_model_config 
            and self.target_model_config.hf_text_config.model_type == "openpangu_vl"
        ):
            self.model = self.target_model_config.model
        elif self.method in ("ngram", "[ngram]"):
            self.model = "ngram"
        else:
            raise ValueError("num_speculative_tokens was provided without " "speculative model.")

    # Automatically configure the method for ngram when "model" is used
    # instead of "method"
    if self.method is None and (self.model is not None and self.model in ("ngram", "[ngram]")):
        self.method = "ngram"

    if self.method in ("ngram", "[ngram]"):
        # Unified to "ngram" internally
        self.method = "ngram"
        # Set default values if not provided
        if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
            # TODO(woosuk): Tune these values. They are arbitrarily chosen.
            self.prompt_lookup_min = 5
            self.prompt_lookup_max = 5
        elif self.prompt_lookup_min is None:
            if self.prompt_lookup_max is None:
                raise ValueError("self.prompt_lookup_max is None")
            self.prompt_lookup_min = self.prompt_lookup_max
        elif self.prompt_lookup_max is None:
            if self.prompt_lookup_min is None:
                raise ValueError("self.prompt_lookup_min is None")
            self.prompt_lookup_max = self.prompt_lookup_min

        # Validate values
        if self.prompt_lookup_min < 1:
            raise ValueError(f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
        if self.prompt_lookup_max < 1:
            raise ValueError(f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                f"prompt_lookup_min={self.prompt_lookup_min} must " f"be <= prompt_lookup_max={self.prompt_lookup_max}"
            )

        self.draft_model_config = self.target_model_config
        self.draft_parallel_config = self.target_parallel_config
    else:
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0

        if self.model is not None:
            self.draft_model_config = ModelConfig(
                model=self.model,
                task="draft",
                tokenizer=self.target_model_config.tokenizer,
                tokenizer_mode=self.target_model_config.tokenizer_mode,
                trust_remote_code=self.target_model_config.trust_remote_code,
                allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                dtype=self.target_model_config.dtype,
                seed=self.target_model_config.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                tokenizer_revision=self.target_model_config.tokenizer_revision,
                spec_target_max_model_len=self.target_model_config.max_model_len,
                quantization=self.quantization,
                enforce_eager=self.target_model_config.enforce_eager,
                max_seq_len_to_capture=self.target_model_config.max_seq_len_to_capture,
                max_logprobs=self.target_model_config.max_logprobs,
                hf_overrides=SpeculativeConfig.hf_config_override,
            )

            # Automatically detect the method
            if self.method in ("eagle", "eagle3"):
                pass
            elif (
                "eagle-" in self.draft_model_config.model.lower() or "eagle3-" in self.draft_model_config.model.lower()
            ):
                self.method = "eagle"
            elif self.draft_model_config.hf_config.model_type == "medusa":
                self.method = "medusa"
            elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                self.method = "mlp_speculator"
            elif self.draft_model_config.hf_config.model_type == "deepseek_mtp":
                self.method = "deepseek_mtp"
            else:
                self.method = "draft_model"

            # Replace hf_config for EAGLE draft_model
            if self.method in ("eagle", "eagle3"):
                if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                    raise ValueError("Chunked prefill and EAGLE are not compatible " "when using V0.")

                from vllm.transformers_utils.configs.eagle import EAGLEConfig

                if isinstance(self.draft_model_config.hf_config, EAGLEConfig):
                    pass
                else:
                    eagle_config = EAGLEConfig(
                        self.draft_model_config.hf_config,
                        method=self.method,
                        model_type="eagle",
                    )
                    self.draft_model_config.hf_config = eagle_config

            if self.num_speculative_tokens is not None and hasattr(
                self.draft_model_config.hf_config, "num_lookahead_tokens"
            ):
                self.draft_model_config.hf_config.num_lookahead_tokens = self.num_speculative_tokens

            n_predict = getattr(self.draft_model_config.hf_config, "n_predict", None)
            if n_predict is not None:
                if self.num_speculative_tokens is None:
                    # Default to max value defined in draft model config.
                    self.num_speculative_tokens = n_predict
                elif self.num_speculative_tokens > n_predict and self.num_speculative_tokens % n_predict != 0:
                    # Ensure divisibility for MTP module reuse.
                    raise ValueError(
                        f"num_speculative_tokens:{self.num_speculative_tokens}" f" must be divisible by {n_predict=}"
                    )

            self.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(
                self.target_parallel_config,
                self.draft_tensor_parallel_size,
                self.draft_model_config.hf_config,
            )

            self.draft_model_config.max_model_len = SpeculativeConfig._maybe_override_draft_max_model_len(
                self.max_model_len,
                self.draft_model_config.max_model_len,
                self.target_model_config.max_model_len,
            )

            self.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
                self.target_parallel_config, self.draft_tensor_parallel_size
            )

    if self.acceptance_method == "typical_acceptance_sampler":
        if self.posterior_threshold is None:
            self.posterior_threshold = 0.09
        if self.posterior_alpha is None:
            self.posterior_alpha = 0.3

    self._verify_args()


@register("vllm.config.ModelConfig.get_diff_sampling_param")
def get_diff_sampling_param(self) -> dict[str, Any]:
    """
    This method returns a dictionary containing the parameters
    that differ from the default sampling parameters. If
    `generation_config` is `"vllm"`, an empty dictionary is returned.

    Returns:
        dict[str, Any]: A dictionary with the differing sampling
        parameters, if `generation_config` is `"vllm"` an empty dictionary.
    """
    if self.generation_config == "vllm":
        config = {}
    else:
        config = self.try_get_generation_config()

    # Overriding with given generation config
    config.update(self.override_generation_config)

    available_params = [
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "max_new_tokens",
        "top_n_sigma",
    ]
    if any(p in config for p in available_params):
        diff_sampling_param = {
            p: config.get(p)
            for p in available_params if config.get(p) is not None
        }
        # Huggingface definition of max_new_tokens is equivalent
        # to vLLM's max_tokens
        if "max_new_tokens" in diff_sampling_param:
            diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                "max_new_tokens")
    else:
        diff_sampling_param = {}

    if diff_sampling_param:
        logger.warning_once(
            "Default sampling parameters have been overridden by the "
            "model's Hugging Face generation config recommended from the "
            "model creator. If this is not intended, please relaunch "
            "vLLM instance with `--generation-config vllm`.")
    return diff_sampling_param