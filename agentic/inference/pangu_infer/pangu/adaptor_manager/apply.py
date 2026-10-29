import logging
import sys
import os

from pangu_infer.pangu.adaptor_manager import adaptor_manager
from pangu_infer.pangu.adaptor_manager.adaptor_manager import set_config, apply_patches


def _setup_logging() -> logging.Logger:
    """Create a stdout logger for this adaptor."""
    logger = logging.getLogger("adaptor_loader")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


log = _setup_logging()

# Align adaptor_manager logger with ours
if not adaptor_manager.glogger.handlers:
    for h in log.handlers:
        adaptor_manager.glogger.addHandler(h)
adaptor_manager.glogger.setLevel(logging.WARNING)


def start_apply_patches(allowed_types: list = None):
    """Run patching for modules"""
    apply_patches(
        patch_package=["pangu_infer.patches.vllm", "pangu_infer.patches.vllm_ascend"],
        allowed_types=["standard"] if allowed_types is None else allowed_types,
    )

