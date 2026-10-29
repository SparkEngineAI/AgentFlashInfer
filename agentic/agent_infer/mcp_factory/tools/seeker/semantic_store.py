import logging
import re
import os
import json
from pathlib import Path
from functools import wraps

import aiofiles
import asyncio
from nltk.tokenize import sent_tokenize
import nltk

# 下载必要的NLTK资源
logger = logging.getLogger(__name__)
from config.config import NLTK_DATA_PATH

nltk.data.path.insert(0, NLTK_DATA_PATH)


def time_cost_decorator(func):
    @wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # 执行原函数并获取返回值
        return result  # 返回原函数的返回值

    return wrapper


async def read_txt_file(file_path: str) -> str:
    """读取Tex文件内容"""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
        return content
    except FileNotFoundError:
        logger.error(f"错误：文件 {file_path} 未找到")
        return ""
    except Exception as e:
        logger.error(f"读取文件时出错：{e}")
        return ""


class TextChunker:
    """
    一个用于对文本进行智能分块的工具类。
    支持中文/英文自动检测，URL 去除，重叠分块。
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30):
        """
        初始化分块器。
        :param chunk_size: 每个块的最大字符数（默认 300）
        :param chunk_overlap: 块之间的重叠字符数（默认 30）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 中文字符正则
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
        # 英文字符正则
        self.english_pattern = re.compile(r'[a-zA-Z]')
        # URL 正则（http/https/ftp/www）
        self.url_pattern = re.compile(r'https?://\S+|ftp://\S+|www\.\S+')

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None, language: str = None) -> list[
        str]:
        """
        对输入文本进行智能分块。
        :param text: 要切分的原始文本。
        :param chunk_size: 每个块的大小（字符数），默认使用初始化值。
        :param chunk_overlap: 块之间的重叠长度，默认使用初始化值。
        :param language: 手动指定语言 'chinese' 或 'english'，若为 None 则自动检测。
        :return: 分块后的字符串列表。
        """
        text = self.remove_urls(text)

        if language is None:
            language = self.detect_language_optimized(text, threshold=0.1)
            if language == 'chinese':
                chunk_size = chunk_size or 400
                chunk_overlap = chunk_overlap or 40
            else:
                chunk_size = chunk_size or 600
                chunk_overlap = chunk_overlap or 60
        else:
            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap

        sentences = self.split_sentences(text, language)

        return self._do_chunking(sentences, chunk_size, chunk_overlap)

    def remove_urls(self, text: str) -> str:
        """去除文本中的 URL"""
        return self.url_pattern.sub('', text)

    def detect_language_optimized(self, text: str, threshold: float = 0.5) -> str:
        """基于字符频率快速判断语言"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        chinese_chars = self.chinese_pattern.findall(text)
        english_chars = self.english_pattern.findall(text)

        total_count = len(chinese_chars) + len(english_chars)
        if total_count == 0:
            return 'english'

        chinese_ratio = len(chinese_chars) / total_count
        return 'chinese' if chinese_ratio > threshold else 'english'

    def split_sentences(self, text: str, language: str) -> list[str]:
        """根据语言选择分句方式"""
        if language == 'chinese':
            return [s.strip() for s in re.split(r'([。！？]|\n)', text) if s.strip()]
        else:
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception as e:
                return [s.strip() for s in re.split(r'([.!?]|\n)', text) if s.strip()]

    def _do_chunking(self, sentences: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        """执行核心分块逻辑"""
        ans_chunks = []
        current_chunk = []
        current_len = 0
        max_chunk = int(1.2 * chunk_size)

        for sent in sentences:
            sent_len = len(sent)

            if sent_len > max_chunk:
                for i in range(0, len(sent), max_chunk):
                    ans_chunks.append(sent[i:i + max_chunk])
                last_one = ans_chunks.pop()
                current_chunk = [last_one]
                current_len = len(last_one)
                continue

            if current_len + sent_len > chunk_size and current_chunk:
                ans_chunks.append(' '.join(current_chunk))
                overlap_ratio = chunk_overlap / chunk_size
                overlap_count = max(1, int(len(current_chunk) * overlap_ratio))
                current_chunk = current_chunk[-overlap_count:]
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sent)
            current_len += sent_len

        if current_chunk:
            ans_chunks.append(' '.join(current_chunk))

        return ans_chunks


class FileMonitor:
    def __init__(self, folder_path, state_file=None):
        self.folder_path = Path(folder_path).resolve()  # # 转为绝对路径
        self.current_state = {}
        self.previous_state = {}
        self.changes = []
        if state_file is None:
            self.state_file = Path(__file__).parent / "file_state.json"
        else:
            self.state_file = Path(state_file).resolve() / "file_state.json"

    def _scan_files_sync(self):
        current_state = {}
        # 定义需要忽略的文件后缀（不带点）
        ignore_exts = {'db', 'json', 'jsonl', 'milvus_demo', 'log'}

        for root, dirs, files in os.walk(self.folder_path):
            root_path = Path(root)
            if self.state_file.name in files and root_path == self.state_file.parent:
                continue
            for file in files:
                file_path = root_path / file

                if file_path == self.state_file:
                    continue
                suffix = file_path.suffix.lstrip('.').lower()
                if not suffix or suffix in ignore_exts:
                    continue
                try:
                    stat = file_path.stat()
                    rel_path = str(file_path.relative_to(self.folder_path))
                    current_state[rel_path] = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'ctime': stat.st_ctime
                    }
                except (OSError, FileNotFoundError):
                    continue
        return current_state

    async def scan_files(self):
        """异步封装扫描操作"""
        self.current_state = await asyncio.to_thread(self._scan_files_sync)

    async def save_state(self):
        """保存当前文件状态到JSON文件"""
        try:
            # aiofiles.open 异步打开文件
            async with aiofiles.open(self.state_file, 'w', encoding='utf-8') as f:
                # json.dump 不支持直接写入 async file object，所以先 dumps 成字符串
                content = json.dumps(self.current_state, indent=2)
                await f.write(content)
        except Exception as e:
            print(f"Error saving state file: {e}")

    async def load_previous_state(self):
        """从JSON文件加载之前保存的文件状态"""
        if self.state_file.exists():
            try:
                async with aiofiles.open(self.state_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    self.previous_state = json.loads(content)
                return True
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            except Exception as e:
                logger.error(f"Error reading state file: {e}")
        return False

    async def compare_states(self):
        """比较当前状态和之前状态，找出变动的文件"""
        self.changes = []
        # 检查新增或修改的文件
        for file, current_info in self.current_state.items():
            if file not in self.previous_state:
                self.changes.append(('created', file))
            else:
                # 2. 检查修改文件 (增加逻辑：如果大小或mtime变化)
                prev_info = self.previous_state[file]
                if (current_info['mtime'] != prev_info['mtime'] or
                        current_info['size'] != prev_info['size']):
                    self.changes.append(('modified', file))


async def get_files(monitor_path, workspace_path=None):
    """folder_path = "/path/to/jina_reader_demo"
    print(GetFiles(folder_path))"""
    # 获取要监视的文件夹路径（默认为当前目录）
    folder_path = monitor_path  # "/path/to/jina_reader_demo"

    # 创建监视器实例
    if workspace_path:
        monitor = FileMonitor(folder_path, state_file=workspace_path)
    else:
        monitor = FileMonitor(folder_path)

    await monitor.scan_files()

    has_previous_state = await  monitor.load_previous_state()

    if not has_previous_state:
        await monitor.save_state()
        return_file = []
        for file in monitor.current_state.keys():
            return_file.append(file)
        return return_file
    else:
        # 后续运行，比较变化
        await monitor.compare_states()
        return_comp = []
        if monitor.changes:
            for change_type, file in monitor.changes:
                return_comp.append(file)  # 添加了修改/创建的文件。
        # 保存新的状态
        await monitor.save_state()
        return return_comp
