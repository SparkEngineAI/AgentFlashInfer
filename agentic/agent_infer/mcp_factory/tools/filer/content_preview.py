import os
import json
from ..tool_register import tool_info_decorator
from typing import Any, Dict
from pathlib import Path
import re
import config.config as config
from .file_stats import file_stats


def _analyze_file_structure(file_path: Path, file_type: str) -> Dict[str, Any]:
    """Analyze file structure based on file type"""

    structure = {
        'file_type': file_type,
        'structure_elements': [],
        'summary': ''
    }
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    if file_type == 'python_code':
        # Analyze Python structure
        import_lines = re.findall(
            r'^(import .*|from .* import .*)', content, re.MULTILINE)
        class_lines = re.findall(r'^class\s+(\w+).*:', content, re.MULTILINE)
        function_lines = re.findall(
            r'^def\s+(\w+)\s*\(', content, re.MULTILINE)

        structure['structure_elements'] = [
            {'type': 'imports', 'count': len(
                import_lines), 'items': import_lines[:10]},
            {'type': 'classes', 'count': len(
                class_lines), 'items': class_lines},
            {'type': 'functions', 'count': len(
                function_lines), 'items': function_lines[:20]}
        ]
        structure['summary'] = f"Python file with {len(class_lines)} classes, {len(function_lines)} functions"

    elif file_type == 'json_data':
        # Analyze JSON structure
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                structure['structure_elements'] = [
                    {'type': 'object_keys', 'count': len(
                        data.keys()), 'items': list(data.keys())[:20]}
                ]
                structure['summary'] = f"JSON object with {len(data.keys())} top-level keys"
            elif isinstance(data, list):
                structure['structure_elements'] = [
                    {'type': 'array_length', 'count': len(data), 'items': []}
                ]
                structure['summary'] = f"JSON array with {len(data)} items"
        except:
            structure['summary'] = "JSON file (structure analysis failed)"

    elif file_type == 'markdown_document':
        # Analyze markdown structure
        headers = re.findall(r'^(#{1,6})\s+(.*)', content, re.MULTILINE)
        links = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)

        structure['structure_elements'] = [
            {'type': 'headers', 'count': len(headers), 'items': [
                f"{h[0]} {h[1]}" for h in headers[:20]]},
            {'type': 'links', 'count': len(links), 'items': links[:10]}
        ]
        structure['summary'] = f"Markdown document with {len(headers)} headers, {len(links)} links"

    elif file_type == 'csv_data':
        # Analyze CSV structure
        lines = content.split('\n')
        if lines:
            first_line = lines[0]
            columns = first_line.split(',')
            structure['structure_elements'] = [
                {'type': 'columns', 'count': len(columns), 'items': [
                    col.strip().strip('"\'') for col in columns]},
                {'type': 'rows', 'count': len(lines) - 1, 'items': []}
            ]
            structure['summary'] = f"CSV file with {len(columns)} columns, {len(lines)-1} data rows"

    else:
        # Generic text analysis
        lines = content.split('\n')
        words = content.split()
        structure['structure_elements'] = [
            {'type': 'lines', 'count': len(lines), 'items': []},
            {'type': 'words', 'count': len(words), 'items': []}
        ]
        structure['summary'] = f"Text file with {len(lines)} lines, {len(words)} words"

    return structure


@tool_info_decorator
def content_preview(
    file_path: str,
    session_id: str,
    preview_lines: int = 20,
    include_structure: bool = True,
    smart_sampling: bool = True
) -> Any:
    """
    Get intelligent preview of file content without reading the entire file.
    Uses smart sampling to give representative content overview.

    Args:
        file_path: Path to the file
        session_id: The conversation ID
        preview_lines: Number of lines to include in preview
        include_structure: Whether to analyze file structure
        smart_sampling: Whether to use intelligent sampling vs just first N lines
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    full_path = workspace_path / file_path
    if not full_path.is_file():
        raise Exception(f"file '{full_path}' don't exist.")
    # Get basic file stats first
    stats = file_stats(file_path, session_id)['data']

    preview_content = {
        'file_path': file_path,
        'file_stats': stats,
        'preview_method': 'simple' if not smart_sampling else 'smart_sampling',
        'content_sections': []
    }

    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        if smart_sampling and stats['line_count'] > preview_lines * 2:
            # Smart sampling: beginning, middle, end
            lines_per_section = preview_lines // 3

            # Beginning
            beginning_lines = []
            for i, line in enumerate(f):
                if i >= lines_per_section:
                    break
                beginning_lines.append({
                    'line_number': i + 1,
                    'content': line.rstrip(),
                    'section': 'beginning'
                })

            # Middle (seek to approximate middle)
            f.seek(0)
            all_lines = f.readlines()
            total_lines = len(all_lines)

            middle_start = (total_lines // 2) - (lines_per_section // 2)
            middle_end = middle_start + lines_per_section

            middle_lines = []
            for i in range(max(0, middle_start), min(total_lines, middle_end)):
                middle_lines.append({
                    'line_number': i + 1,
                    'content': all_lines[i].rstrip(),
                    'section': 'middle'
                })

            # End
            end_lines = []
            end_start = max(0, total_lines - lines_per_section)
            for i in range(end_start, total_lines):
                end_lines.append({
                    'line_number': i + 1,
                    'content': all_lines[i].rstrip(),
                    'section': 'end'
                })

            preview_content['content_sections'] = [
                {'section_name': 'Beginning', 'lines': beginning_lines},
                {'section_name': 'Middle', 'lines': middle_lines},
                {'section_name': 'End', 'lines': end_lines}
            ]

        else:
            # Simple preview: just first N lines
            simple_lines = []
            for i, line in enumerate(f):
                if i >= preview_lines:
                    break
                simple_lines.append({
                    'line_number': i + 1,
                    'content': line.rstrip(),
                    'section': 'preview'
                })

            preview_content['content_sections'] = [
                {'section_name': 'Preview', 'lines': simple_lines}
            ]

    # Structure analysis
    if include_structure:
        structure = _analyze_file_structure(full_path, stats['file_type'])
        preview_content['structure_analysis'] = structure

    return preview_content
