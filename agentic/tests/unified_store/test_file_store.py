# tests/agentic_rl/unified_store/test_filestore_integration.py

import pytest
import ray
from pathlib import Path
from agent_infer.unified_store.unified_store import UnifiedStoreManager, get_store

import shutil

def cleanup_dir(path):
    # 删除目录下的所有内容，但不删除目录本身
    for child in Path(path).iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

@pytest.fixture(scope="module", autouse=True)
def setup_ray_and_file_store():
    # 启动 Ray 并启动文件存储服务
    ray.init()
    mgr = UnifiedStoreManager.remote()
    root = ray.get(mgr.start_file_store.remote())
    yield root
    ray.shutdown()
    shutil.rmtree(root, ignore_errors=True)


def test_file_store_basic_crud(setup_ray_and_file_store):
    root = setup_ray_and_file_store
    store = get_store()
    store.add_store(["file"])

    # Create and read
    filepath = "dir1/test.txt"
    store.file.create(filepath, "hello")
    full = Path(root) / filepath
    assert full.exists()
    assert store.file.read(filepath) == "hello"

    # Write (overwrite) and read
    store.file.write(filepath, "world")
    assert store.file.read(filepath) == "world"

    # Append and read
    store.file.append(filepath, "!!!")
    assert store.file.read(filepath) == "world!!!"

    # Delete and verify removal
    store.file.delete(filepath)
    assert not full.exists()

    cleanup_dir(root)


def test_file_store_list_files(setup_ray_and_file_store):
    root = setup_ray_and_file_store
    store = get_store()
    store.add_store(["file"])

    # Set up hierarchy
    store.file.create("a.txt", "A")
    store.file.create("subdir/b.txt", "B")

    # Non-recursive listing
    flat = store.file.list_files("", recursive=False)
    names_flat = {info.name for info in flat}
    assert names_flat == {"a.txt", "subdir"}

    # Recursive listing
    recursive = store.file.list_files("", recursive=True)
    names_rec = {info.name for info in recursive}
    assert names_rec >= {"a.txt", "b.txt"}

    # Hidden files should be excluded by default
    hidden_path = ".hidden"
    store.file.create(hidden_path, "H")
    listed = store.file.list_files("", recursive=True)
    assert all(info.name != ".hidden" for info in listed)

    # Include hidden files
    listed_all = store.file.list_files("", recursive=True, include_hidden=True)
    names_all = {info.name for info in listed_all}
    assert ".hidden" in names_all

    cleanup_dir(root)


def test_file_store_errors(setup_ray_and_file_store):
    root = setup_ray_and_file_store
    store = get_store()
    store.add_store(["file"])

    # Reading non-existent file
    with pytest.raises(Exception):
        store.file.read("no_such.txt")

    # Deleting non-existent file
    with pytest.raises(Exception):
        store.file.delete("no_such.txt")

    # Listing non-existent directory
    with pytest.raises(Exception):
        store.file.list_files("no_dir")

    # Listing a file path as directory
    store.file.create("file_only.txt", "data")
    with pytest.raises(Exception):
        store.file.list_files("file_only.txt")

    cleanup_dir(root)

