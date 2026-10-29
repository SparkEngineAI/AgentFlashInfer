"""
Auto-import all modules under the current package to trigger registration decorators or expose functions.
"""

import pkgutil
import importlib

__all__ = []

# Recursively walk this package and import all modules
for _, module_name, _ in pkgutil.walk_packages(path=__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)
    # Expose module names if needed
    __all__.append(module_name.split(".")[-1])
