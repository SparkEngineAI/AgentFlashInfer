import os
from typing import List
from ..tool_register import tool_info_decorator
from typing import Any
from pathlib import Path
import config.config as config
import re


@tool_info_decorator
def file_grep_search(
    pattern: str,
    session_id: str,
    file_pattern: str = "*",
    recursive: bool = True,
    ignore_case: bool = False,
    max_matches: int = 100
) -> Any:
    """Search for pattern in files using grep-like functionality"""
    import fnmatch

    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)

    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    matches = []
    search_path = workspace_path

    def _search_file(file_path: Path):
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append({
                    'file': str(file_path.relative_to(workspace_path)),
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'match_start': regex.search(line).start() if regex.search(line) else 0
                })

                if len(matches) >= max_matches:
                    return False  # Stop searching

        return True

    # Search files
    if recursive:
        for file_path in search_path.rglob(file_pattern):
            if file_path.is_file():
                if not _search_file(file_path):
                    break
    else:
        for file_path in search_path.glob(file_pattern):
            if file_path.is_file():
                if not _search_file(file_path):
                    break
    if not matches:
        raise Exception("No matching content was found")
    return matches


@tool_info_decorator
def file_find_by_name(
    name_pattern: str,
    session_id: str,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 100
) -> Any:
    """Find files by name pattern"""
    import fnmatch

    if not case_sensitive:
        name_pattern = name_pattern.lower()

    matches = []
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    search_path = workspace_path

    def _match_name(file_path: Path) -> bool:
        name = file_path.name
        if not case_sensitive:
            name = name.lower()

        return fnmatch.fnmatch(name, name_pattern)

    # Search files
    if recursive:
        for file_path in search_path.rglob("*"):
            if _match_name(file_path):
                matches.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(workspace_path)),
                    'type': 'directory' if file_path.is_dir() else 'file',
                    'size': file_path.stat().st_size if file_path.is_file() else None
                })

                if len(matches) >= max_results:
                    break
    else:
        for file_path in search_path.iterdir():
            if _match_name(file_path):
                matches.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(workspace_path)),
                    'type': 'directory' if file_path.is_dir() else 'file',
                    'size': file_path.stat().st_size if file_path.is_file() else None
                })

                if len(matches) >= max_results:
                    break
    if not matches:
        raise Exception("No matching content was found")
    return matches


@tool_info_decorator
def file_grep_with_context(
    pattern: str,
    session_id: str,
    file_path: str = None,
    file_pattern: str = "*",
    recursive: bool = True,
    ignore_case: bool = False,
    context_before: int = 2,
    context_after: int = 2,
    max_matches: int = 50
) -> Any:
    """
    Enhanced grep with context lines before and after matches.
    Much more useful for understanding match context.

    Args:
        pattern: Regex pattern to search for
        session_id: The conversation ID
        file_path: Specific file to search (if provided, ignores file_pattern)
        file_pattern: File pattern to search in (ignored if file_path provided)
        recursive: Whether to search recursively
        ignore_case: Whether to ignore case
        context_before: Number of lines to include before each match
        context_after: Number of lines to include after each match
        max_matches: Maximum number of matches to return
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)
    import fnmatch

    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)

    matches = []

    # Determine files to search
    files_to_search = []
    if file_path:
        # Search specific file
        full_path = workspace_path / file_path
        if full_path.exists() and full_path.is_file():
            files_to_search.append(full_path)
    else:
        # Search by pattern
        search_path = workspace_path
        if recursive:
            for file_path in search_path.rglob(file_pattern):
                if file_path.is_file():
                    files_to_search.append(file_path)
        else:
            for file_path in search_path.glob(file_pattern):
                if file_path.is_file():
                    files_to_search.append(file_path)

    for file_path in files_to_search:
        if len(matches) >= max_matches:
            break

        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines()

        # Find all matching lines
        matching_line_nums = []
        for line_num, line in enumerate(lines):
            if regex.search(line):
                matching_line_nums.append(line_num)

        # Extract context for each match
        for line_num in matching_line_nums:
            if len(matches) >= max_matches:
                break

            # Calculate context range
            start_line = max(0, line_num - context_before)
            end_line = min(len(lines), line_num + context_after + 1)

            # Extract context lines
            context_lines = []
            for i in range(start_line, end_line):
                line_info = {
                    'line_number': i + 1,
                    'content': lines[i],
                    'is_match': i == line_num,
                    'match_positions': []
                }

                # Find match positions in the matching line
                if i == line_num:
                    for match in regex.finditer(lines[i]):
                        line_info['match_positions'].append({
                            'start': match.start(),
                            'end': match.end(),
                            'matched_text': match.group()
                        })

                context_lines.append(line_info)

            match_info = {
                'file': str(file_path.relative_to(workspace_path)),
                'match_line_number': line_num + 1,
                'match_content': lines[line_num],
                'context_lines': context_lines,
                'context_start_line': start_line + 1,
                'context_end_line': end_line,
                'total_context_lines': len(context_lines)
            }

            matches.append(match_info)

    return matches


@tool_info_decorator
def file_read_lines(
    file_path: str,
    session_id: str,
    start_line: int = 1,
    end_line: int = None,
    max_lines: int = 1000
) -> Any:
    """
    Read specific line ranges from a file without loading the entire file.
    Perfect for reading specific sections after grep or for large files.

    Args:
        file_path: Path to the file
        session_id: The conversation ID
        start_line: Starting line number (1-based)
        end_line: Ending line number (1-based, None for end of file)
        max_lines: Maximum number of lines to read (safety limit)
    """

    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    full_path = workspace_path / file_path

    lines_read = []
    current_line = 0

    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            current_line += 1

            # Skip lines before start_line
            if current_line < start_line:
                continue

            # Stop if we've reached end_line
            if end_line and current_line > end_line:
                break

            # Safety check for max_lines
            if len(lines_read) >= max_lines:
                break

            lines_read.append({
                'line_number': current_line,
                'content': line.rstrip('\n\r')
            })

    # Calculate actual end line
    actual_end_line = lines_read[-1]['line_number'] if lines_read else start_line - 1

    return {
        'file_path': file_path,
        'start_line': start_line,
        'end_line': actual_end_line,
        'lines': lines_read,
        'line_count': len(lines_read)
    }


@tool_info_decorator
def concat_files(
    file_paths: List[str],
    output_file: str,
    session_id: str,
    add_separators: bool = False,
    separator: str = "\n---\n",
    create_dirs: bool = True
) -> Any:
    """
    Concatenate multiple files into a single output file.

    Args:
        file_paths: List of file paths to concatenate (in order)
        output_file: Path where the concatenated content will be written
        session_id: The conversation ID
        add_separators: Whether to add separators between files
        separator: Separator string to use between files (if add_separators=True)
        create_dirs: Whether to create parent directories for output file

    Returns:
        MCPToolResult indicating success or failure of the operation
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    # Prepare output path
    output_path = workspace_path / output_file

    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track file processing
    concatenated_content = []
    processed_files = []
    missing_files = []
    total_size = 0

    for file_path in file_paths:
        full_path = workspace_path / file_path

        if not full_path.exists():
            missing_files.append(file_path)
            continue

        # Read file content
        content = full_path.read_text(encoding='utf-8')

        # Ensure content ends with newline to avoid joining lines
        if content and not content.endswith('\n'):
            content += '\n'

        concatenated_content.append(content)
        processed_files.append({
            'file_path': file_path,
            'size_bytes': len(content.encode('utf-8')),
            'line_count': len(content)
        })
        total_size += len(content.encode('utf-8'))

    # Join content with optional separators
    if add_separators and len(concatenated_content) > 1:
        final_content = separator.join(concatenated_content)
    else:
        final_content = ''.join(concatenated_content)

    # Write concatenated content to output file
    output_path.write_text(final_content, encoding='utf-8')

    # Create success message
    success_message = f"Successfully concatenated {len(processed_files)} files into {output_file}"
    if missing_files:
        success_message += f". Warning: {len(missing_files)} files were not found and skipped: {missing_files}"

    return success_message
