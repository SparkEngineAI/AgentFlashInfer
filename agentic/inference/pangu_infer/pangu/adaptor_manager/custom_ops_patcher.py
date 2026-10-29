# hook_install.py
from __future__ import annotations

import ast
from typing import Callable, Iterable, Optional, Set, Tuple
import importlib.machinery as mach

_INSTALLED: bool = False
_STRIPPED_PATHS: Set[str] = set()
_ORIG_GET_CODE: Optional[Callable] = None


def install_unified_attention_register_strip_hook(
    *,
    target_module: str = "vllm_ascend.attention.attention_v1",
    target_op: str = "unified_ascend_attention_with_output",
    blacklist_substrings: Optional[Iterable[str]] = (
        "/patches/",
        "\\patches\\",
        "pangu_infer/patches",
        "pangu_infer\\patches",
    ),
    marker_text: str = "RUNNING OLD",
    verbose: bool = True,
) -> None:
    """
    Install an AST-based import hook that removes a specific top-level
    `direct_register_custom_op(op_name=...)` call from the target module at import time.
    Also attempts to remove an adjacent marker `print(...)` placed right above the call.

    Args:
        target_module (str): Fully-qualified module name to patch.
        target_op (str): The `op_name` string to match in `direct_register_custom_op`.
        blacklist_substrings (Optional[Iterable[str]]): Path substrings that, if present
            in the loader's file path, will skip patching (to avoid touching your own
            shadow/patch files).
        marker_text (str): Text marker used to identify the marker `print(...)`.
            Any `print(...)` arg containing this text—or a call to `*.print_stack()`—is removed.
        verbose (bool): If True, prints diagnostic messages.

    Returns:
        None
    """
    global _INSTALLED, _ORIG_GET_CODE

    if _INSTALLED:
        if verbose:
            print("[hook] Already installed; skipping re-install.", flush=True)
        return

    _INSTALLED = True
    _ORIG_GET_CODE = mach.SourceFileLoader.get_code  # type: ignore[attr-defined]
    blist = tuple(blacklist_substrings or ())

    def _is_marker_print(node: ast.AST) -> bool:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            return False
        call = node.value
        is_print = (
            (isinstance(call.func, ast.Name) and call.func.id == "print")
            or (isinstance(call.func, ast.Attribute) and call.func.attr == "print")
        )
        if not is_print:
            return False

        def _has_marker(x: ast.AST) -> bool:
            if isinstance(x, ast.Constant) and isinstance(x.value, str) and marker_text in x.value:
                return True
            if isinstance(x, ast.Call):
                f = x.func
                if isinstance(f, ast.Attribute) and f.attr == "print_stack":
                    return True
            return False

        return any(_has_marker(a) for a in call.args) or any(
            _has_marker(kw.value) for kw in (call.keywords or [])
        )

    def _is_target_register(node: ast.AST) -> bool:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            return False
        call = node.value
        func_name = (
            call.func.id
            if isinstance(call.func, ast.Name)
            else call.func.attr
            if isinstance(call.func, ast.Attribute)
            else None
        )
        if func_name != "direct_register_custom_op":
            return False

        op_name: Optional[str] = None
        for kw in (call.keywords or []):
            if (
                kw.arg == "op_name"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                op_name = kw.value.value
                break
        if op_name is None and call.args:
            a0 = call.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                op_name = a0.value
        return op_name == target_op

    def _strip_register_and_marker(src: str, filename: str) -> Tuple[ast.AST, int, int]:
        if verbose:
            print(f"[hook] Parsing source: {filename}", flush=True)
        tree = ast.parse(src, filename=filename)
        body = list(tree.body)
        to_delete = set()
        removed_reg = 0
        removed_mrk = 0

        def snippet(node: ast.AST) -> str:
            try:
                seg = ast.get_source_segment(src, node)
                return (seg if seg is not None else "").strip()
            except Exception:
                return f"<{type(node).__name__}>"

        for i, node in enumerate(body):
            if _is_target_register(node):
                to_delete.add(i)
                removed_reg += 1

                # Look back up to 3 statements for a marker print
                found_marker = False
                for back in (1, 2, 3):
                    j = i - back
                    if j < 0:
                        break
                    cand = body[j]
                    if _is_marker_print(cand):
                        to_delete.add(j)
                        removed_mrk += 1
                        found_marker = True
                        if verbose:
                            print("[hook] >> Delete marker print + register", flush=True)
                        break
                if not found_marker and verbose:
                    print("[hook] >> Delete register only (no nearby marker)", flush=True)

        if to_delete:
            tree.body = [n for k, n in enumerate(body) if k not in to_delete]
            ast.fix_missing_locations(tree)

        if verbose:
            print(f"[hook] Removal done: register={removed_reg}, marker={removed_mrk}", flush=True)
        return tree, removed_reg, removed_mrk  # type: ignore[return-value]

    def _patched_get_code(self, fullname: str):  # type: ignore[override]
        if fullname != target_module:
            return _ORIG_GET_CODE(self, fullname)  # type: ignore[misc]

        path = getattr(self, "path", None)
        path_str = str(path) if path is not None else ""
        path_norm = path_str.replace("\\", "/")

        if any(s in path_norm for s in blist):
            if verbose:
                print(f"[hook] Skip (blacklist): {fullname} @ {path_str}", flush=True)
            return _ORIG_GET_CODE(self, fullname)  # type: ignore[misc]

        if path_str in _STRIPPED_PATHS:
            if verbose:
                print(f"[hook] Skip (already processed): {fullname} @ {path_str}", flush=True)
            return _ORIG_GET_CODE(self, fullname)  # type: ignore[misc]

        if verbose:
            print(f"[hook] Hit: {fullname} @ {path_str}", flush=True)

        src = self.get_source(fullname)
        if verbose:
            print(f"[hook] Source length={len(src)}", flush=True)

        tree, n_reg, n_mark = _strip_register_and_marker(src, path_str or fullname)
        code = compile(tree, path_str or fullname, "exec")

        if verbose:
            if n_reg > 0:
                print(f"[hook] ✅ Stripped {n_reg} register(s), {n_mark} marker print(s).", flush=True)
            else:
                print(f"[hook] ⚠️ No target register found (op_name={target_op}).", flush=True)

        _STRIPPED_PATHS.add(path_str)
        return code

    mach.SourceFileLoader.get_code = _patched_get_code  # type: ignore[assignment]
    if verbose:
        print(f"[hook] Installed AST strip hook: module={target_module}, op={target_op}", flush=True)
