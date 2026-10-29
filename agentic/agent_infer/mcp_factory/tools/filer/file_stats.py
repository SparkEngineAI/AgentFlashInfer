from ..tool_register import tool_info_decorator
import logging
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv
import config.config as config
import os
load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
def file_stats(file_path: str, session_id: str) -> Any:
    """
    Get comprehensive file statistics without reading full content.
    Perfect for deciding whether to read full file or use targeted extraction.

    Args:
        file_path: Path to the file (relative to workspace)
        session_id: The conversation ID
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)
    full_path = workspace_path / file_path

    # Get basic file stats
    stat_info = full_path.stat()
    file_size = stat_info.st_size

    # Quick content analysis without loading full file
    encoding = 'utf-8'
    line_count = 0
    word_count = 0
    char_count = 0
    first_lines = []
    last_lines = []

    with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
        # Read first few lines for preview
        for i, line in enumerate(f):
            line_count += 1
            if i < 5:  # First 5 lines
                first_lines.append(line.rstrip())

            char_count += len(line)
            word_count += len(line.split())

            # For efficiency, stop detailed counting after reasonable limit
            if line_count > 10000:
                # Estimate remaining based on average
                remaining_size = file_size - f.tell()
                if remaining_size > 0:
                    avg_line_size = f.tell() / line_count
                    estimated_remaining_lines = int(
                        remaining_size / avg_line_size)
                    line_count += estimated_remaining_lines

                    # Estimate words and chars
                    avg_chars_per_line = char_count / min(line_count, 10000)
                    avg_words_per_line = word_count / min(line_count, 10000)
                    char_count += int(remaining_size)
                    word_count += int(estimated_remaining_lines *
                                      avg_words_per_line)
                break

    # Get last few lines if file is reasonable size
    if file_size < 1024 * 1024:  # Less than 1MB
        with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
            last_lines = [line.rstrip() for line in lines[-5:]]
            if line_count <= 10000:  # Recalculate if we estimated
                line_count = len(lines)
                char_count = sum(len(line) for line in lines)
                word_count = sum(len(line.split()) for line in lines)

    # Determine file type
    file_extension = full_path.suffix.lower()
    file_type = _detect_file_type(full_path, file_extension)

    # Reading recommendation
    reading_recommendation = _get_reading_recommendation(
        file_size, line_count, word_count, file_type
    )

    stats = {
        'file_path': file_path,
        'file_size_bytes': file_size,
        'file_size_human': _format_file_size(file_size),
        'line_count': line_count,
        'word_count': word_count,
        'character_count': char_count,
        'encoding': encoding,
        'file_type': file_type,
        'file_extension': file_extension,
        'modified_time': stat_info.st_mtime,
        'is_large_file': file_size > 1024 * 1024,  # > 1MB
        'is_very_large_file': file_size > 10 * 1024 * 1024,  # > 10MB
        'first_lines_preview': first_lines,
        'last_lines_preview': last_lines,
        'reading_recommendation': reading_recommendation
    }
    result_dict = {
        "data": stats,
        "metadata": {
            'analysis_method': 'efficient_sampling' if line_count > 10000 else 'full_analysis'
        }
    }
    return result_dict


def _detect_file_type(file_path: Path, extension: str) -> str:
    """Detect file type based on extension and content"""

    # Extension-based detection
    type_map = {
        '.py': 'python_code',
        '.js': 'javascript_code',
        '.ts': 'typescript_code',
        '.java': 'java_code',
        '.cpp': 'cpp_code',
        '.c': 'c_code',
        '.html': 'html_markup',
        '.css': 'css_stylesheet',
        '.json': 'json_data',
        '.xml': 'xml_data',
        '.yaml': 'yaml_config',
        '.yml': 'yaml_config',
        '.md': 'markdown_document',
        '.txt': 'plain_text',
        '.csv': 'csv_data',
        '.sql': 'sql_code',
        '.sh': 'shell_script',
        '.dockerfile': 'docker_config',
        '.env': 'environment_config'
    }

    if extension in type_map:
        return type_map[extension]

    # Content-based detection for unknown extensions
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()

            if first_line.startswith('#!'):
                return 'executable_script'
            elif first_line.startswith('<?xml'):
                return 'xml_data'
            elif first_line.startswith('{') or first_line.startswith('['):
                return 'json_data'
            elif 'DOCTYPE html' in first_line or '<html' in first_line:
                return 'html_markup'
    except:
        pass

    return 'unknown_text'


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _get_reading_recommendation(file_size: int, line_count: int,
                                word_count: int, file_type: str) -> Dict[str, Any]:
    """Provide intelligent recommendations for how to read the file"""

    recommendations = {
        'strategy': 'full_read',
        'reason': 'File is small enough for full reading',
        'alternatives': []
    }

    # Large file strategies
    if file_size > 1024 * 1024:  # > 1MB
        recommendations['strategy'] = 'selective_read'
        recommendations['reason'] = 'File is large, consider targeted approaches'
        recommendations['alternatives'] = [
            'Use file_grep_with_context to search for specific content',
            'Use content_preview to get overview before full read',
            'Use file_read_lines to read specific sections',
            'Use semantic_search if content is indexed'
        ]

    elif line_count > 1000:
        recommendations['strategy'] = 'preview_first'
        recommendations['reason'] = 'Many lines, preview recommended before full read'
        recommendations['alternatives'] = [
            'Use content_preview for quick overview',
            'Use file_grep_with_context for specific searches'
        ]

    # File type specific recommendations
    if file_type in ['json_data', 'xml_data']:
        recommendations['alternatives'].append(
            'Consider parsing structure instead of full text read')
    elif file_type.endswith('_code'):
        recommendations['alternatives'].append(
            'Use grep to find specific functions/classes')
    elif file_type == 'csv_data':
        recommendations['alternatives'].append(
            'Consider reading headers first with file_read_lines')

    return recommendations
