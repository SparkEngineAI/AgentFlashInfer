"""
sitecustomize hook for vLLM patching.

This module:
1) Detects whether the current process looks like a vLLM Worker.
2) If yes, loads vLLM plugins and applies Pangu patches.
3) If the multiprocessing is set to spawn, reset pangu config

Place this file as `sitecustomize.py` on PYTHONPATH so it auto-runs.
"""

from __future__ import annotations

import os
import sys
import json
import logging


def _looks_like_worker_process() -> bool:
    """Detect if this is a vLLM Worker process."""
    try:
        import psutil

        cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
        print("Multiprocessing calling:", cmdline, flush=True)
        if "from multiprocessing" in cmdline:
            return True
    except Exception as e:
        raise e

    return False


IS_WORKER: bool = _looks_like_worker_process()
IS_SPAWN: bool = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "") == "spawn"

if IS_WORKER:

    # Separate adaptor manager's logger
    from pangu_infer.pangu.adaptor_manager import adaptor_manager
    glogger = adaptor_manager.glogger
    level_name = os.getenv("ADAPTOR_LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.ERROR)
    if not glogger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(logging.Formatter(
            fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        glogger.addHandler(h)

    glogger.propagate = False
    glogger.setLevel(level)

    # Read pangu config and set
    payload = os.environ.get("PANGU_CONFIG_JSON")
    if payload:
        from pangu_infer.pangu.adaptor_manager.adaptor_manager import (
            set_config,
            apply_patches,
        )

        cfg = json.loads(payload)
        set_config(cfg)


    # pangu infer patch
    from pangu_infer.models import vllm_ascend

    # Plugins need to be loaded at the engine/scheduler level too
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    from pangu_infer.pangu.adaptor_manager.apply import start_apply_patches

    # only patch standard patches, avoid patch any worker patches as there is a bug in setDevice.
    start_apply_patches(allowed_types=["standard"])
