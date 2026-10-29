import os

from vllm.config import ModelConfig
from vllm import ModelRegistry
from vllm_ascend import models

from pangu_infer.pangu.adaptor_manager.apply import start_apply_patches


vllm_ascend_register_model = models.register_model


def register_models():
    """Register Pangu models after vllm-ascend's models to override models with same names"""

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "pangu_infer.models.vllm_ascend.modeling_openpangu_embedded:PanguEmbeddedForCausalLM",
    )

    # trigger patch apply after vllm ascend is initialized
    start_apply_patches()

    from pangu_infer.patches.vllm.config import is_deepseek_mla

    # patch @property
    ModelConfig.is_deepseek_mla = is_deepseek_mla


models.register_model = register_models
