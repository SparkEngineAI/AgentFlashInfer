# adaptor_loader.py
import sys
import json

from pangu_infer.pangu.adaptor_manager import adaptor_manager
from pangu_infer.pangu.adaptor_manager.adaptor_manager import set_config, apply_patches

import logging, sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S"
))

adaptor_manager.glogger.addHandler(handler)
adaptor_manager.glogger.setLevel(logging.DEBUG)


# Load your config however you like; e.g., via env var or CLI arg
cfg = json.loads(sys.argv[1])  # or os.environ["PATCH_CONFIG_JSON"]
set_config(cfg)

from pangu_infer.models import vllm_ascend

apply_patches(patch_package=["pangu_infer.patches.vllm", "pangu_infer.patches.vllm_ascend"])

# Remove the first argument so vLLM doesn’t see it
#    sys.argv[0] stays as the script name, and we drop sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]

# start vllm process
from vllm.entrypoints.cli.main import main as vllm_main
sys.exit(vllm_main())
