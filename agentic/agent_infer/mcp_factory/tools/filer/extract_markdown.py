import os
from typing import List, Dict, Optional
from ..tool_register import tool_info_decorator
from typing import Any
import re
import os
from pathlib import Path
import config.config as config


def _detect_header_patterns(line: str, line_num: int, max_depth: int = 6) -> Optional[Dict[str, Any]]:
    """
    Detect various header patterns in markdown files.
    Supports:
    1. Traditional markdown headers: # Header, ## Header, etc.
    2. Bold text headers: **Header Text**
    3. Numbered bold headers: **4.1 Section Title**
    4. Label-style headers: **Why it matters:** content

    Args:
        line: The line to analyze
        line_num: Line number for reference
        max_depth: Maximum depth to consider

    Returns:
        Dict with header info if detected, None otherwise
    """
    line_stripped = line.strip()

    if not line_stripped:
        return None

    # Pattern 1: Traditional markdown headers (# ## ### etc.)
    if line_stripped.startswith('#'):
        level = 0
        for char in line_stripped:
            if char == '#':
                level += 1
            else:
                break

        if level <= max_depth:
            header_text = line_stripped[level:].strip()
            if header_text:
                return {
                    'level': level,
                    'text': header_text,
                    'line_number': line_num,
                    'anchor_id': _create_anchor_id(header_text),
                    'markdown_syntax': '#' * level,
                    'header_type': 'traditional',
                    'full_line': line
                }

    # Pattern 2: Bold text headers **Header Text**
    bold_pattern = re.match(r'^\*\*([^*]+)\*\*\s*$', line_stripped)
    if bold_pattern:
        header_text = bold_pattern.group(1).strip()
        # Determine level based on content patterns
        level = _determine_bold_header_level(header_text)

        if level <= max_depth:
            return {
                'level': level,
                'text': header_text,
                'line_number': line_num,
                'anchor_id': _create_anchor_id(header_text),
                'markdown_syntax': f'**{header_text}**',
                'header_type': 'bold',
                'full_line': line
            }

    # Pattern 3: Bold text with following content **Label:** content
    label_pattern = re.match(r'^\*\*([^*]+):\*\*\s*(.*)$', line_stripped)
    if label_pattern:
        header_text = label_pattern.group(1).strip()
        following_content = label_pattern.group(2).strip()

        # These are typically section labels, treat as level 3-4 headers
        level = 4 if len(header_text.split()) <= 3 else 3

        if level <= max_depth:
            return {
                'level': level,
                'text': header_text,
                'line_number': line_num,
                'anchor_id': _create_anchor_id(header_text),
                'markdown_syntax': f'**{header_text}:**',
                'header_type': 'label',
                'full_line': line,
                'following_content': following_content
            }

    return None


def _determine_bold_header_level(header_text: str) -> int:
    """
    Determine the hierarchy level of a bold header based on content patterns.

    Args:
        header_text: The text inside **bold** markers

    Returns:
        Estimated level (1-6)
    """
    # Pattern detection for level determination
    text_lower = header_text.lower()

    # Level 1: Main titles (long, no numbering, important keywords)
    main_keywords = ['introduction', 'conclusion',
                     'summary', 'overview', 'executive summary']
    if any(keyword in text_lower for keyword in main_keywords) and len(header_text.split()) >= 3:
        return 1

    # Level 2: Major sections with numbering like "4. The Titans of Industry"
    if re.match(r'^\d+\.\s+', header_text) and len(header_text.split()) >= 3:
        return 2

    # Level 3: Subsections with decimal numbering like "4.1 Tesla Model 3"
    if re.match(r'^\d+\.\d+\s+', header_text):
        return 3

    # Level 4: Sub-subsections with more complex numbering like "4.1.1 Technical Details"
    if re.match(r'^\d+\.\d+\.\d+\s+', header_text):
        return 4

    # Level 2: Numbered sections without decimal
    if re.match(r'^\d+\s+', header_text) and len(header_text.split()) >= 2:
        return 2

    # Level 3: Short descriptive headers (2-4 words)
    if 2 <= len(header_text.split()) <= 4:
        return 3

    # Level 2: Longer descriptive headers (5+ words)
    if len(header_text.split()) >= 5:
        return 2

    # Default: Level 3 for other cases
    return 3


def _create_anchor_id(header_text: str) -> str:
    """Create GitHub-style anchor ID from header text"""
    # Convert to lowercase, replace spaces and special chars with hyphens
    anchor = re.sub(r'[^\w\s-]', '', header_text.lower())
    anchor = re.sub(r'[\s_-]+', '-', anchor)
    return anchor.strip('-')


def _build_toc_tree(headers: List[Dict]) -> List[Dict]:
    """Build hierarchical tree structure from flat header list"""
    if not headers:
        return []

    tree = []
    stack = []  # Stack to keep track of parent headers

    for header in headers:
        header_copy = header.copy()
        header_copy['children'] = []

        # Pop stack until we find the appropriate parent level
        while stack and stack[-1]['level'] >= header['level']:
            stack.pop()

        if stack:
            # Add as child to the last header in stack
            stack[-1]['children'].append(header_copy)
        else:
            # Top-level header
            tree.append(header_copy)

        stack.append(header_copy)

    return tree


def _generate_markdown_toc(headers: List[Dict]) -> str:
    """Generate markdown-formatted table of contents"""
    toc_lines = []

    for header in headers:
        indent = '  ' * (header['level'] - 1)
        line_info = f" (Line {header['line_number']})" if header['line_number'] else ""
        toc_line = f"{indent}- [{header['text']}](#{header['anchor_id']}){line_info}"
        toc_lines.append(toc_line)

    return '\n'.join(toc_lines)


def _generate_indented_toc(headers: List[Dict]) -> str:
    """Generate simple indented text table of contents"""
    toc_lines = []

    for header in headers:
        indent = '  ' * (header['level'] - 1)
        line_info = f" (Line {header['line_number']})" if header['line_number'] else ""
        toc_line = f"{indent}{header['markdown_syntax']} {header['text']}{line_info}"
        toc_lines.append(toc_line)

    return '\n'.join(toc_lines)


def _generate_navigation_links(headers: List[Dict], file_path: str) -> List[Dict]:
    """Generate navigation links for each header"""
    nav_links = []

    for header in headers:
        nav_link = {
            'text': header['text'],
            'level': header['level'],
            'file_link': f"{file_path}#{header['anchor_id']}",
            'line_number': header['line_number'],
            'anchor_id': header['anchor_id']
        }
        nav_links.append(nav_link)

    return nav_links


@tool_info_decorator
def extract_markdown_toc(
    file_path: str,
    session_id: str,
    max_depth: int = 6,
    include_line_numbers: bool = True,
    sort_by: str = "document_order"
) -> Any:
    """
    Extract all headers from a markdown file and create a structured table of contents.

    Args:
        file_path: Path to the markdown file
        session_id: The conversation ID
        max_depth: Maximum header depth to include (1-6, where 1 = #, 6 = ######)
        include_line_numbers: Whether to include line numbers for each header
        sort_by: How to sort headers - "document_order", "alphabetical", or "level_then_alpha"
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    full_path = workspace_path / file_path

    content = full_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    headers = []

    # Extract headers with line numbers using enhanced detection
    for line_num, line in enumerate(lines, 1):
        header_info = _detect_header_patterns(line, line_num, max_depth)

        if header_info:
            # Add line number based on settings
            if not include_line_numbers:
                header_info['line_number'] = None

            headers.append(header_info)

    # Sort headers based on sort_by parameter
    if sort_by == "alphabetical":
        headers.sort(key=lambda h: h['text'].lower())
    elif sort_by == "level_then_alpha":
        headers.sort(key=lambda h: (h['level'], h['text'].lower()))
    # "document_order" keeps the original order (no sorting needed)

    # Create hierarchical structure
    toc_tree = _build_toc_tree(headers)

    # Generate different TOC formats
    toc_formats = {
        'structured_list': headers,
        'hierarchical_tree': toc_tree,
        'markdown_toc': _generate_markdown_toc(headers),
        'indented_text': _generate_indented_toc(headers),
        'navigation_links': _generate_navigation_links(headers, file_path)
    }

    # Statistics
    level_counts = {}
    for header in headers:
        level = header['level']
        level_counts[level] = level_counts.get(level, 0) + 1

    return {
        'file_path': file_path,
        'total_headers': len(headers),
        'level_distribution': level_counts,
        'deepest_level': max(h['level'] for h in headers) if headers else 0,
        'toc_formats': toc_formats,
        'extraction_settings': {
            'max_depth': max_depth,
            'include_line_numbers': include_line_numbers,
            'sort_by': sort_by
        }
    }


@tool_info_decorator
def extract_markdown_section(
    file_path: str,
    header_title: str,
    session_id: str,
    exact_match: bool = False,
    include_subsections: bool = True,
    include_header: bool = True,
    max_depth: int = 6
) -> Any:
    """
    Extract content from a specific section of a markdown file based on header title.

    Args:
        file_path: Path to the markdown file
        header_title: Title of the header section to extract
        session_id: The conversation ID
        exact_match: If True, requires exact title match; if False, allows partial matching
        include_subsections: If True, includes all subsections under the target header
        include_header: If True, includes the header line itself in the extracted content
        max_depth: Maximum header depth to consider when stopping extraction
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    full_path = workspace_path / file_path

    content = full_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Find the target header using enhanced detection
    target_header_info = None
    target_line_index = -1

    for line_num, line in enumerate(lines):
        header_info = _detect_header_patterns(line, line_num + 1, max_depth)

        if header_info:
            header_text = header_info['text']

            # Check for match
            if exact_match:
                match = header_text.lower() == header_title.lower()
            else:
                match = header_title.lower() in header_text.lower()

            if match:
                target_header_info = {
                    'level': header_info['level'],
                    'text': header_text,
                    'line_number': line_num + 1,
                    'line_index': line_num,
                    'full_line': line,
                    'header_type': header_info['header_type'],
                    'markdown_syntax': header_info['markdown_syntax']
                }
                target_line_index = line_num
                break

    # Extract section content
    section_lines = []
    current_index = target_line_index

    # Include the header line if requested
    if include_header:
        section_lines.append(lines[current_index])
        current_index += 1
    else:
        current_index += 1  # Skip the header line

    # Extract content until next header of same or higher level
    target_level = target_header_info['level']

    while current_index < len(lines):
        line = lines[current_index]

        # Check if this is a header using enhanced detection
        header_info = _detect_header_patterns(
            line, current_index + 1, max_depth)

        if header_info:
            header_level = header_info['level']

            # Stop if we hit a header of same or higher level (unless including subsections)
            if header_level <= target_level:
                break

            # If not including subsections, stop at any header
            if not include_subsections and header_level > target_level:
                break

        section_lines.append(line)
        current_index += 1

    # Join the extracted content
    extracted_content = '\n'.join(section_lines)

    # Calculate content statistics
    word_count = len(extracted_content.split())
    line_count = len(section_lines)
    char_count = len(extracted_content)

    # Find any subsections included
    subsections = []
    if include_subsections:
        for line_num, line in enumerate(section_lines):
            header_info = _detect_header_patterns(
                line, line_num + 1, max_depth)

            if header_info and header_info['level'] > target_level:
                subsections.append({
                    'level': header_info['level'],
                    'text': header_info['text'],
                    'relative_level': header_info['level'] - target_level,
                    'header_type': header_info['header_type'],
                    'markdown_syntax': header_info['markdown_syntax']
                })

    return {
        'file_path': file_path,
        'extracted_content': extracted_content,
        'header_info': target_header_info,
        'content_stats': {
            'line_count': line_count,
            'word_count': word_count,
            'character_count': char_count
        },
        'subsections_found': subsections,
        'extraction_settings': {
            'header_title_searched': header_title,
            'exact_match': exact_match,
            'include_subsections': include_subsections,
            'include_header': include_header,
            'max_depth': max_depth
        }
    }
