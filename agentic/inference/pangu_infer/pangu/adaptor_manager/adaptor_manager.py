import os
import json
import importlib
import pkgutil
import logging
from collections import OrderedDict
from pangu_infer.pangu.utils.pangu_config import PanguConfig

# Configure module-level logger
glogger = logging.getLogger(__name__)

# Global registry mapping target paths to patch callables
_registry = OrderedDict()

# Types of functions
_func_types = OrderedDict()

# Global configuration storage for patches
_config = None


def set_config(config: dict):
    """
    Set the global configuration for patch functions as a PanguConfig instance.

    Converts the input dict into a PanguConfig, so that all registered patches
    can access config values via attribute access.

    Args:
        config (dict): Dictionary of configuration parameters.
    """
    global _config
    _config = PanguConfig(config)
    glogger.debug("Global PanguConfig set: %s", _config)


def get_config() -> PanguConfig:
    """
    Retrieve the global PanguConfig instance.

    Returns:
        PanguConfig: The configuration object, or None if not set.
    """
    global _config
    if _config is None:
        # Some vLLM processes may be launched without sitecustomize
        # This avoids import errors for some get_config() calls
        env_config = os.environ.get("PANGU_CONFIG_JSON", None)
        if env_config is not None:
            env_config = json.loads(env_config)
            set_config(env_config)
    return _config


def register(target_path: str, patch_type: str = "standard"):
    """
    Decorator to register a patch for a specified target path.

    Args:
        target_path (str): Fully qualified path of the target attribute
            (e.g., 'module.submodule.Class.method').

    Raises:
        RuntimeError: If a patch for the same target_path has already been registered.
    """

    def decorator(func):
        if target_path in _registry:
            glogger.warning("Patch for %s already registered, skipping.", target_path)
            return _registry[target_path]  # Return the existing function if it's already registered
        glogger.debug("Register patch: %s -> %s.%s", target_path, func.__module__, func.__name__)
        _registry[target_path] = func
        _func_types[target_path] = patch_type
        return func

    return decorator


def _import_all_patch_modules(package: str):
    """
    Recursively import all modules under the given package to trigger register decorators.

    Args:
        package (str): Name of the package to scan (e.g., 'patches').
    """
    pkg = importlib.import_module(package)
    for _, mod_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod_name)
        if is_pkg:
            _import_all_patch_modules(mod_name)


def apply_patches(patch_package: list = None, allowed_types: list = None):
    """
    Apply all registered patches by monkey-patching target attributes.

    Optionally scans a package for modules with @register decorators,
    then dynamically replaces original attributes (functions, methods, classes)
    with the registered patch implementations.

    Args:
        patch_package (list, optional): List of names of the package to scan for patch modules.
            If None, scanning is skipped.
    """
    glogger.info("Starting patch application process...")

    if allowed_types is None:
        allowed_types = ["standard", "worker"]

    # Scan for patch modules if provided
    if patch_package:
        for p in patch_package:
            glogger.debug("Scanning patch package '%s' for modules...", p)
            _import_all_patch_modules(p)

    applied_count = 0

    # Apply each registered patch in the order registered
    for target_path, patch_callable in _registry.items():
        if _func_types[target_path] not in allowed_types:
            glogger.debug(
                "Skipped not allowed patch: %s type: %s",
                target_path,
                _func_types[target_path],
            )
            continue

        glogger.debug(
            "Processing registered patch: %s -> %s.%s",
            target_path,
            patch_callable.__module__,
            patch_callable.__name__,
        )
        parts = target_path.split(".")

        # Identify the deepest module path
        module = None
        attrs = []
        for i in range(len(parts), 0, -1):
            module_name = ".".join(parts[:i])
            try:
                module = importlib.import_module(module_name)
                attrs = parts[i:]
                glogger.debug(
                    "Imported module '%s', attributes to resolve: %s",
                    module_name,
                    attrs,
                )
                break
            except ImportError:
                continue

        if module is None:
            error_msg = f"Cannot import module part of target path '{target_path}'"
            glogger.error(error_msg)
            raise ImportError(error_msg)

        # Traverse to the parent of the final attribute
        parent = module
        for attr in attrs[:-1]:
            if not hasattr(parent, attr):
                error_msg = f"Object '{parent}' has no attribute '{attr}' to patch"
                glogger.error(error_msg)
                raise AttributeError(error_msg)
            parent = getattr(parent, attr)
            glogger.debug("Navigated to parent object: %s", parent)

        # The attribute to be replaced or added
        final_attr = attrs[-1] if attrs else None
        if not final_attr:
            error_msg = f"No attribute specified in target path '{target_path}'"
            glogger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve the existing attribute on the parent, if any, and log the action
        existing = getattr(parent, final_attr, None)
        if existing:
            glogger.info(
                "Replacing '%s' defined in %s.%s with patch %s.%s",
                final_attr,
                existing.__module__,
                existing.__name__,
                patch_callable.__module__,
                patch_callable.__name__,
            )
        else:
            glogger.info(
                "Adding new attribute '%s' to %s via patch %s.%s",
                final_attr,
                parent,
                patch_callable.__module__,
                patch_callable.__name__,
            )

        # Apply the patch to the resolved parent object
        setattr(parent, final_attr, patch_callable)
        applied_count += 1

        # For any modules that still hold a direct reference to the old object,
        # replace that reference with the new patch_callable
        if existing:
            import sys

            for mod in list(sys.modules.values()):
                # Only swap out attributes that exactly match the original object
                if getattr(mod, final_attr, None) is existing and mod != patch_callable.__module__:
                    glogger.info(
                        "Propagating the module changes '%s' to %s via patch %s.%s",
                        final_attr,
                        mod,
                        patch_callable.__module__,
                        patch_callable.__name__,
                    )
                    setattr(mod, final_attr, patch_callable)

    glogger.info("Patch application complete. Total patches applied: %d", applied_count)
