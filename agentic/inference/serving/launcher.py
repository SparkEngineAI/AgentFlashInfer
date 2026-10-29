"""
Adaptor loader for vLLM + Pangu patching.

Features
--------
1) Parse CLI into a dict (top-level keys are kebab-case, e.g., 'trust-remote-code').
2) Parse --additional-config as JSON (also accepts JSON-like input where keys
   are unquoted; e.g., {foo:1, bar:{x:true}}).
3) Read CFG__* environment overrides (supports nesting with '__'):
   - Top-level CFG keys: '_' -> '-' to match vLLM CLI.
   - Nested path under additional-config is NOT hyphenized.
4) Log useful information during CLI/env parsing, merging, and final argv build.
5) BEFORE launching vLLM:
   - Convert TOP-LEVEL hyphenated keys to snake_case (kebab '-' -> underscore '_')
     for PanguConfig, e.g., 'additional-config' -> 'additional_config'.
   - Register that dict via `set_config`, and apply patches.
6) Finally, rebuild argv for vLLM (still kebab-case) and invoke vLLM CLI main.

Flag rule
---------
For boolean flags (e.g., trust-remote-code):
- True  -> emit '--flag'
- False -> omit the flag
- Bare presence (no value) and empty env value mean True.
"""

from __future__ import annotations
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# ----------------------------- Logging ----------------------------- #

def _setup_logging() -> logging.Logger:
    """Create a stdout logger for this adaptor."""
    logger = logging.getLogger("launcher")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

log = _setup_logging()

# ----------------------------- Conversions ----------------------------- #

TRUE_SET = {"true"}
FALSE_SET = {"false"}


def to_bool(value: str | None) -> bool | None:
    """Return bool if value looks like a boolean literal; None otherwise.
    None is treated as True for bare flags.
    """
    if value is None:
        return True
    s = value.strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return None


def coerce_scalar(value: str | None) -> Any:
    """Convert to bool/int/float when possible; otherwise return original string.
    Bare flags (None) become True.
    """
    b = to_bool(value)
    if b is not None:
        return b
    if value is None:
        return True
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        try:
            return int(value)
        except Exception:
            logger.warning("Convert to INT ERROR")
    try:
        return float(value)
    except Exception:
        return value


def hyphenize_top(key: str) -> str:
    """Convert '_' to '-' at the top-level to match vLLM CLI style."""
    return key.replace("_", "-")


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst."""
    for k, v in src.items():
        if isinstance(dst.get(k), dict) and isinstance(v, dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set d[keys[0]]...[keys[-1]] = value, creating intermediate dicts."""
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


# ----------------------------- JSON helpers ----------------------------- #

_JSON_KEY_RE = re.compile(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_.\-]*)\s*:')

def _normalize_json_like(s: str) -> str:
    """Best-effort transform from JSON-like to strict JSON:
    - Quote unquoted object keys (foo: -> "foo":).
    - Convert single quotes to double quotes.
    """
    s2 = _JSON_KEY_RE.sub(r'\1"\2":', s)
    s2 = s2.replace("'", '"')
    return s2


def parse_json_loose(s: str) -> Any:
    """Parse JSON; if it fails, try normalizing JSON-like input."""
    try:
        return json.loads(s)
    except Exception:
        s2 = _normalize_json_like(s)
        return json.loads(s2)  # may raise if still invalid


def _collect_braced_json(tokens: List[str], start: int) -> Tuple[str, int]:
    """Collect tokens into a single JSON string starting at index 'start'
    using brace depth. Returns (json_str, last_index_consumed).
    """
    buf: List[str] = []
    depth = 0
    i = start
    while i < len(tokens):
        t = tokens[i]
        buf.append(t)
        depth += t.count("{")
        depth -= t.count("}")
        if depth <= 0 and "}" in t:
            break
        i += 1
    return " ".join(buf), i


# ----------------------------- CLI parsing ----------------------------- #

def parse_cli(argv: List[str]) -> Tuple[str, List[str], Dict[str, Any]]:
    """Parse CLI into (subcommand, positionals, params).
    Notes:
      - Top-level keys are kebab-case.
      - Bare flags -> True.
      - --additional-config accepts strict JSON or JSON-like input.
      - Supports '--key=value'.
    """
    if not argv:
        print("Usage: python new_adaptor_loader.py <subcommand> [...]", file=sys.stderr)
        sys.exit(2)

    subcommand = argv[0]
    rest = argv[1:]

    positionals: List[str] = []
    params: Dict[str, Any] = {}

    log.info("CLI subcommand: %s", subcommand)

    i = 0
    while i < len(rest):
        tok = rest[i]

        # Support --key=value
        if tok.startswith("--") and "=" in tok:
            key_raw, val_raw = tok[2:].split("=", 1)
            key = hyphenize_top(key_raw)
            if key in ("additional-config", "additional_config"):
                params["additional-config"] = parse_json_loose(val_raw)
                log.info("CLI: --%s (json) parsed", key)
            else:
                params[key] = coerce_scalar(val_raw)
                log.info("CLI: --%s = %r", key, params[key])
            i += 1
            continue

        if tok.startswith("--"):
            key_raw = tok[2:]
            key = hyphenize_top(key_raw)

            nxt = rest[i + 1] if i + 1 < len(rest) else None
            if key in ("additional-config", "additional_config") and nxt and nxt.startswith("{"):
                json_str, j = _collect_braced_json(rest, i + 1)
                params["additional-config"] = parse_json_loose(json_str)
                log.info("CLI: --%s (json) collected across tokens", key)
                i = j + 1
                continue

            val: str | None = None
            if nxt is not None and not nxt.startswith("--"):
                val = nxt
                i += 1

            if key == "additional-config" and val is not None:
                params[key] = parse_json_loose(val)
                log.info("CLI: --%s (json) parsed", key)
            else:
                params[key] = coerce_scalar(val)
                log.info("CLI: --%s = %r", key, params[key])
        else:
            positionals.append(tok)
            log.info("CLI positional: %s", tok)

        i += 1

    return subcommand, positionals, params


# ----------------------------- Env overrides ----------------------------- #

def parse_env_overrides() -> Dict[str, Any]:
    """Build a params dict from CFG__* environment variables."""
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith("CFG__"):
            continue
        path = k[5:]
        if not path:
            continue

        parts = path.split("__")
        top = hyphenize_top(parts[0])
        raw_val = v
        val = None if v == "" else v  # empty -> True

        if top in ("additional-config", "additional_config"):
            if len(parts) == 1:
                if val is None:
                    payload: Any = True
                else:
                    try:
                        payload = parse_json_loose(val)
                    except Exception:
                        payload = coerce_scalar(val)
                deep_merge(out, {"additional-config": payload})
                log.info("ENV: %s -> %r (top-level additional-config)", k, payload)
            else:
                sub = out.setdefault("additional-config", {})
                parsed = True if val is None else coerce_scalar(val)
                set_nested(sub, parts[1:], parsed)
                log.info("ENV: %s -> %r (nested additional-config)", k, parsed)
        else:
            parsed = True if val is None else coerce_scalar(val)
            deep_merge(out, {top: parsed})
            log.info("ENV: %s -> %r", k, parsed)

        if raw_val != v:
            log.debug("ENV raw: %s = %r", k, raw_val)

    return out


# ----------------------------- Diff & Rebuild ----------------------------- #

def _diff_top(old: Dict[str, Any], new: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Compute top-level differences for logging."""
    keys = set(old.keys()) | set(new.keys())
    diffs: List[Tuple[str, Any, Any]] = []
    for k in sorted(keys):
        ov, nv = old.get(k, "<MISSING>"), new.get(k, "<MISSING>")
        if ov != nv:
            diffs.append((k, ov, nv))
    return diffs


def rebuild_vllm_argv(subcommand: str, positionals: List[str], params: Dict[str, Any]) -> List[str]:
    """Turn params dict back into vLLM CLI argv."""
    argv: List[str] = ["vllm", subcommand, *positionals]
    keys = [k for k in params.keys() if k != "additional-config"]
    if "additional-config" in params:
        keys.append("additional-config")

    for k in keys:
        v = params[k]
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)  # True -> include
                log.info("EMIT FLAG: %s (True)", flag)
            else:
                log.info("OMIT FLAG: %s (False)", flag)
        elif isinstance(v, dict):
            js = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
            argv.extend([flag, js])
            log.info("ARG JSON: %s = %s", flag, js)
        else:
            argv.extend([flag, str(v)])
            log.info("ARG VAL : %s = %r", flag, v)

    return argv


# ----------------------------- Pangu top-key mapping ----------------------------- #

def to_pangu_top_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert TOP-LEVEL kebab-case keys ('-' to '_') for PanguConfig set_config.
    Example:
      'additional-config' -> 'additional_config'
      'trust-remote-code' -> 'trust_remote_code'
    Nested dicts are left untouched.
    """
    out: Dict[str, Any] = {}
    for k, v in params.items():
        snake = k.replace("-", "_")
        if snake != k:
            log.info("PANGU KEY MAP: %s -> %s", k, snake)
        # If both kebab and snake exist (unlikely), merge dicts; otherwise overwrite.
        if snake in out and isinstance(out[snake], dict) and isinstance(v, dict):
            out[snake] = deep_merge(out[snake], v)
        else:
            out[snake] = v
    return out


# ----------------------------- Main ----------------------------- #

def main() -> None:
    """Entry point: parse, env override, log diffs, set Pangu config (snake keys),
    patch, and call vLLM (kebab keys).
    """
    # 1) Parse CLI
    subcommand, positionals, params_cli = parse_cli(sys.argv[1:])
    log.info("Parsed CLI params (top-level): %s", list(params_cli.keys()))

    # 2) Env overrides
    env_overrides = parse_env_overrides()
    log.info("Parsed ENV overrides (top-level): %s", list(env_overrides.keys()))

    # 3) Merge: ENV overrides CLI
    before = dict(params_cli)
    params = deep_merge(params_cli, env_overrides)
    diffs = _diff_top(before, params)
    if diffs:
        log.info("Overrides applied (top-level diffs):")
        for k, ov, nv in diffs:
            log.info(" - %s: %r -> %r", k, ov, nv)
    else:
        log.info("No top-level overrides applied.")

    # 4) Register final config into PanguConfig (TOP-LEVEL '-' -> '_') and apply patches

    from pangu_infer.pangu.adaptor_manager import adaptor_manager
    from pangu_infer.pangu.adaptor_manager.adaptor_manager import set_config, apply_patches

    # Separate adaptor manager's logger
    level_name = os.getenv("ADAPTOR_LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.ERROR)
    if not adaptor_manager.glogger.handlers:
        for h in log.handlers:
            new_h = logging.StreamHandler(sys.stdout)
            new_h.setLevel(level)
            new_h.setFormatter(h.formatter)
            adaptor_manager.glogger.addHandler(new_h)

    adaptor_manager.glogger.propagate = False
    adaptor_manager.glogger.setLevel(level)

    params_for_pangu = to_pangu_top_keys(params)
    log.info("Pangu set_config keys: %s", list(params_for_pangu.keys()))
    set_config(params_for_pangu)  # <-- now uses snake_case top-level keys

    # Save to env for spawn processes
    os.environ["PANGU_CONFIG_JSON"] = json.dumps(params_for_pangu, ensure_ascii=False)

    # Ensure related models are importable before patching
    # Note that currently we run patching after vllm-ascend is fully initialized
    # So the patching is triggered inside models.__init__
    from pangu_infer.models import vllm_ascend

    # 5) Rebuild argv and invoke vLLM
    vllm_argv = rebuild_vllm_argv(subcommand, positionals, params)
    log.info("Final argv -> vLLM: %s", " ".join(vllm_argv))

    sys.argv = vllm_argv
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.exit(vllm_main())


if __name__ == "__main__":
    main()
