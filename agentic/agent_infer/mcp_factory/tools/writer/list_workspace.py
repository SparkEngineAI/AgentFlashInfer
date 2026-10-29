import asyncio

from ..tool_register import tool_info_decorator
import logging
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
import config.config as config
import os
load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
async def list_workspace(
    session_id: str,
    path: str = None,
    recursive: bool = False,
    include_hidden: bool = False,
    max_depth: int = 3
) -> Any:
    """
    List files and directories in workspace with tree structure visualization

    Args:
        session_id: The conversation ID
        path: Specific path to list (relative to session workspace)
        recursive: Whether to list recursively
        include_hidden: Whether to include hidden files
        max_depth: Maximum recursion depth
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)
    target_path = workspace_path

    items = []
    tree_structure = []

    def _list_items(current_path: Path, current_depth: int = 0):
        if current_depth > max_depth:
            return

        # Get all items and sort them (directories first, then files, both alphabetically)
        all_items = list(current_path.iterdir())
        if not include_hidden:
            all_items = [
                item for item in all_items if not item.name.startswith('.')]

        # Sort: directories first, then files, both alphabetically
        all_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

        for item in all_items:
            item_info = {
                'name': item.name,
                'path': str(item.relative_to(workspace_path)),
                'type': 'directory' if item.is_dir() else 'file',
                'size': item.stat().st_size if item.is_file() else None,
                'modified': item.stat().st_mtime,
                'depth': current_depth
            }

            items.append(item_info)

            # Recurse into directories
            if recursive and item.is_dir():
                _list_items(item, current_depth + 1)

    async def _generate_tree_structure(current_path: Path, prefix: str = "", is_last: bool = True, current_depth: int = 0):
        """Generate ASCII tree structure recursively"""
        if current_depth > max_depth:
            return

        try:
            # Get all items and sort them (directories first, then files, both alphabetically)
            all_items = await list_directory_async(current_path ,include_hidden)

            # Sort: directories first, then files, both alphabetically
            all_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for i, item in enumerate(all_items):
                is_last_item = i == len(all_items) - 1

                # Choose the appropriate tree symbols
                if is_last_item:
                    current_symbol = "└── "
                    extension = "    "
                else:
                    current_symbol = "├── "
                    extension = "│   "

                # Add file/directory indicator
                if item.is_dir():
                    name_with_indicator = f"📁 {item.name}/"
                else:
                    # Add file size for files
                    try:
                        size = item.stat().st_size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size/1024:.1f}KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f}MB"
                        name_with_indicator = f"📄 {item.name} ({size_str})"
                    except:
                        name_with_indicator = f"📄 {item.name}"

                tree_line = prefix + current_symbol + name_with_indicator
                tree_structure.append(tree_line)

                # Recurse into directories
                if recursive and item.is_dir():
                    await _generate_tree_structure(
                        item,
                        prefix + extension,
                        is_last_item,
                        current_depth + 1
                    )

        except PermissionError:
            tree_structure.append(prefix + "└── [Permission Denied]")

    # Generate both flat list and tree structure
    _list_items(target_path)

    # Generate tree structure
    root_name = target_path.name if target_path.name else "workspace"
    tree_structure.append(f"📁 {root_name}/")

    if recursive:
        await _generate_tree_structure(target_path)
    else:
        # For non-recursive, just show immediate children
        all_items = await list_directory_async(target_path, include_hidden)
        all_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

        for i, item in enumerate(all_items):
            is_last_item = i == len(all_items) - 1
            symbol = "└── " if is_last_item else "├── "

            if item.is_dir():
                name_with_indicator = f"📁 {item.name}/"
            else:
                try:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    name_with_indicator = f"📄 {item.name} ({size_str})"
                except:
                    name_with_indicator = f"📄 {item.name}"

            tree_structure.append(symbol + name_with_indicator)

    # Create the tree string
    tree_string = "\n".join(tree_structure)

    result_dict = {
        "data": {
            'items': items,
            'tree_structure': tree_string,
            'tree_lines': tree_structure
        },
        "metadata": {
            'target_path': str(target_path.relative_to(workspace_path)) if path else '.',
            'total_items': len(items),
            'recursive': recursive,
            'max_depth': max_depth,
            'include_hidden': include_hidden
        }
    }
    return result_dict


async def list_directory_async(current_path: Path, include_hidden: bool = True):
    # 在线程池中执行阻塞的 iterdir()
    all_items = await asyncio.to_thread(lambda: list(current_path.iterdir()))

    if not include_hidden:
        all_items = [item for item in all_items if not item.name.startswith('.')]

    return all_items