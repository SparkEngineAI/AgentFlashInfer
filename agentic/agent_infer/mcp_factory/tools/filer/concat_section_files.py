import os
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from agent_infer.mcp_factory.tools.tool_register import tool_info_decorator
from config import config

logger = logging.getLogger(__name__)

@tool_info_decorator
def concat_section_files(
        section_files: List[Dict],
        final_file_path: str
) -> Any:
    """
    Concatenate the content of the saved section files into a single file

    Args:
    section_files (List[Dict]): List of section files to concatenate. Each dictionary must contain at least the key 'file_path' (str), Relative path to the saved section file
    final_file_path (str): The final file path to save the concatenated content, save the file in the workspace **under the relative path `./report/`**, and specify the final_file_path as `./report/final_report.md`
    """
    try:
        logger.debug(f"Starting file concatenation: section_files={section_files}, final_file={final_file_path}")

        # Convert relative path to absolute if needed
        final_file_path = _safe_join(final_file_path)

        output_dir = os.path.dirname(final_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 读取section_files中的文件内容
        merge_reports(section_files, final_file_path)
        return {
            "success":True,
            "data":{"merged_files": len(section_files), "output_path": str(final_file_path)},
            "metadata":{
                'final_file_path': str(final_file_path),
                'section_count': len(section_files)
            }
        }
    except Exception as e:
        logger.error(f"Concatenate section files failed: {e}")
        return {
            "success":False,
            "error":str(e)
        }


def _safe_join(path: str) -> Path:
    if os.path.isabs(path):
        raise Exception(f"Path '{path}' is absolute. Only relative paths are allowed.")
    joined_path = os.path.join(config.WORKSPACE_PATH, path)
    full_joined_path = os.path.realpath(joined_path)
    full_workspace_path = os.path.realpath(config.WORKSPACE_PATH)
    if not full_joined_path.startswith(full_workspace_path):
        raise Exception(f"Path '{path}' is outside workspace directory.")
    return Path(full_joined_path)

def merge_reports(section_contents, output_file):
    report_files = []
    for section_content in section_contents:
        # Handle both dict format (expected) and string format (fallback)
        if isinstance(section_content, dict):
            file_path = section_content.get('file_path')
        else:
            # Fallback for direct string paths
            file_path = section_content

        if file_path:
            full_path = _safe_join(file_path)
            report_files.append(full_path)

    # 提取文件名中的数字索引并排序
    def extract_index(file_path):
        """从文件名中提取数字索引"""
        filename = os.path.basename(file_path)
        match = re.search(r'part_(\d+)\.md', filename)
        if match:
            return int(match.group(1))
        return None  # 不符合格式的文件

    # 创建(索引, 文件路径)元组列表并排序
    indexed_files = []
    for file_path in report_files:
        idx = extract_index(file_path)
        if idx is not None:
            indexed_files.append((idx, file_path))

    # 按索引排序
    indexed_files.sort(key=lambda x: x[0])

    if not indexed_files:
        logger.warning("No report_*.md files found matching criteria")
        return

    # 合并文件
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for idx, file_path in indexed_files:
                filename = os.path.basename(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # 写入文件内容
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                        # 添加分隔标记
                        # outfile.write(f"\n\n--- 文件结束: {filename} ---\n\n")
                    logger.info(f"Merged: {filename}")
                except Exception as e:
                    logger.warning(f"Unable to read file {filename}: {str(e)}")

        logger.info(f"\nMerge completed! The result is saved at: {output_file}")
        logger.info(f"A total of {len(indexed_files)} files have been merged.")

    except Exception as e:
        logger.error(f"Error: Failed to write to output file - {str(e)}")
        # 清理可能创建的不完整输出文件
        if os.path.exists(output_file):
            os.remove(output_file)
        raise