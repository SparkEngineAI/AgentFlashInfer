import dataclasses
import os
from pathlib import Path


@dataclasses.dataclass
class FileInfo:
    name: str
    path: str
    type: str
    size: int
    modified: float
    depth: int


class FileStoreClient:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    # create file
    def create(self, file_path: str, content: str = None):
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            if full_path.exists():
                raise Exception(f'File is existed : {full_path}')
            if not full_path.parent.exists():
                full_path.parent.mkdir()
            full_path.write_text(content or '', encoding='utf-8')
        except Exception as e:
            raise e

    def delete(self, file_path: str):
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            if not full_path.exists():
                raise Exception(f'File does not exist: {full_path}')
            full_path.unlink()
        except Exception as e:
            raise e

    def read(self, file_path: str, encoding: str = 'utf-8') -> str:
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            if not full_path.exists():
                raise Exception(f"File does not exist: {file_path}")
            with open(full_path, 'r', encoding=encoding) as file:
                content = file.read()
                return content
        except Exception as e:
            raise e

    def read_lines(self, file_path: str, lint: int = -1, encoding: str = 'utf-8') -> list[str]:
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            with open(full_path, 'r', encoding=encoding) as file:
                contents = file.readlines(lint)
                return contents
        except Exception as e:
            raise e

    def write(self, file_path: str, content: str, encoding: str = 'utf-8', create_dirs: bool = True) -> int:
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding=encoding) as file:
                file.write(content)
            return full_path.stat().st_size
        except Exception as e:
            raise e

    def write_lines(self, file_path: str, contents: list[str], encoding: str = 'utf-8'):
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            with open(full_path, 'w', encoding=encoding) as file:
                file.writelines(contents)
        except Exception as e:
            raise e

    def append(self, file_path: str, content: str, encoding: str = 'utf-8'):
        try:
            full_path = Path(os.path.join(self.root_dir, file_path))
            with open(full_path, 'a', encoding=encoding) as file:
                file.write(content)
        except Exception as e:
            raise e

    def list_files(self, directory: str, recursive: bool = False, max_depth: int = 3, include_hidden: bool = False) -> \
            list[FileInfo]:
        try:
            full_path = Path(os.path.join(self.root_dir, directory))
            if not full_path.exists():
                raise Exception(f"Directory path is not exist: {full_path}")
            if not full_path.is_dir():
                raise Exception(f"Directory path is not a directory: {full_path}")

            file_items = []

            def _list_items(current_path: Path, current_depth: int = 0):
                if current_depth > max_depth:
                    return
                try:
                    for item in current_path.iterdir():
                        # Skip hidden files if not requested
                        if not include_hidden and item.name.startswith('.'):
                            continue

                        file_info = FileInfo(name=item.name, path=str(full_path),
                                             type='directory' if item.is_dir() else 'file',
                                             size=item.stat().st_size if item.is_file() else None,
                                             modified=item.stat().st_mtime,
                                             depth=current_depth)
                        file_items.append(file_info)

                        # Recurse into directories
                        if recursive and item.is_dir():
                            _list_items(item, current_depth + 1)

                except PermissionError:
                    pass  # Skip directories we can't read

            _list_items(full_path)
            return file_items

        except Exception as e:
            raise e

    def file_states(self, file_path: str) -> FileInfo:
        pass
