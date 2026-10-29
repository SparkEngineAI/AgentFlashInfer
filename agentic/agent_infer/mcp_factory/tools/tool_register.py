import config.config as config
from openai import OpenAI, AsyncOpenAI
from pathlib import Path
from typing import Dict, Callable, Any
import time
import functools
import importlib
import inspect
import logging
import pkgutil
import time
import traceback
import asyncio
from typing import Dict, Any


logger = logging.getLogger(__name__)

MODEL_NAME = config.DEFAULT_MODEL_NAME
BASE_URL = config.DEFAULT_MODEL_URL
API_KEY = config.DEFAULT_MODEL_API_KEY
base_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
base_client_async = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def tool_info_decorator(func):
    """
    一个通用的工具处理装饰器，可以同时支持同步和异步函数。
    """

    # --- 针对异步函数的 wrapper ---
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)

            end_time = time.time()
            logger.info(
                f"\n============== Tool Info START (async) ==============")
            logger.info(f"Tool name: {func.__name__}")
            logger.info(f"Tool result: {result}")
            logger.info(f"Execution time: {end_time - start_time:.4f} seconds")
            logger.info(f"============== Tool Info END ==============\n")

            return {
                "success": True,
                "data": result,
                "error": None,
            }
        except Exception as e:
            end_time = time.time()
            logger.info(
                f"\n============== Tool Info START (async) ==============")
            logger.info(f"Tool name: {func.__name__}")
            logger.info(f"Tool result: Error - {str(e)}")
            logger.info(f"Execution time: {end_time - start_time:.4f} seconds")
            logger.info(f"============== Tool Info END ==============\n")

            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "data": None,

            }

    # --- 针对同步函数的 wrapper
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)

            end_time = time.time()
            logger.info(
                f"\n============== Tool Info START (sync) ==============")
            logger.info(f"Tool name: {func.__name__}")
            logger.info(f"Tool result: {result}")
            logger.info(f"Execution time: {end_time - start_time:.4f} seconds")
            logger.info(f"============== Tool Info END ==============\n")

            return {
                "success": True,
                "data": result,
                "error": None,
            }
        except Exception as e:
            end_time = time.time()
            logger.info(
                f"\n============== Tool Info START (sync) ==============")
            logger.info(f"Tool name: {func.__name__}")
            logger.info(f"Tool result: Error - {str(e)}")
            logger.info(f"Execution time: {end_time - start_time:.4f} seconds")
            logger.info(f"============== Tool Info END ==============\n")

            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            return {
                "success": False,
                "error": error_msg,
                "data": None,
            }

    # --- 根据被装饰的函数类型，返回对应的 wrapper ---
    if asyncio.iscoroutinefunction(func):
        async_wrapper._is_tool = True
        return async_wrapper
    else:
        sync_wrapper._is_tool = True
        return sync_wrapper


class ToolScanner:
    def __init__(self, base_package: str):
        self.base_package = base_package
        self.tool_registry: Dict[str, Callable] = {}

    def scan_package(self) -> None:
        """扫描整个Python包结构"""
        package = importlib.import_module(self.base_package)
        package_path = Path(package.__file__).parent

        for finder, name, _ in pkgutil.walk_packages(
            [str(package_path)], prefix=f"{self.base_package}."
        ):
            try:
                module = importlib.import_module(name)
                self._scan_module(module)
            except ImportError as e:
                logger.info(f"Failed to import {name}: {e}")

    def _scan_module(self, module: Any) -> None:
        """扫描单个模块中的工具函数"""
        for name, obj in inspect.getmembers(module):
            if self._is_tool_function(obj):
                self.tool_registry[f"{module.__name__}.{name}"] = obj

    def _is_tool_function(self, obj: Any) -> bool:
        """检查是否为工具函数"""
        return (
            inspect.isfunction(obj)
            and hasattr(obj, "__wrapped__")
            and getattr(obj, "_is_tool", False)
        )

    def get_tools(self) -> Dict[str, Callable]:
        """获取所有注册的工具函数"""
        return self.tool_registry.copy()
