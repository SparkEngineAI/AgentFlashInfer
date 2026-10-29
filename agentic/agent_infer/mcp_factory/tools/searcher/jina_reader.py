import os
from agent_infer.mcp_factory.tool_cache.tool_cache_lib import tool
from ..tool_register import tool_info_decorator
from ..tool_result_format import MCPToolResult
import logging
import requests
from typing import Dict, Any, Union, List, Optional
from dotenv import load_dotenv
from pathlib import Path
from bs4 import BeautifulSoup
import dateutil.parser
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import time
import re
import config.config as config

load_dotenv()

logger = logging.getLogger(__name__)


def _extract_title_from_content(content: str) -> str:
    """Extract title from markdown content"""
    try:
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()

        # Fallback: use first non-empty line, truncated
        for line in lines[:5]:
            line = line.strip()
            if line and not line.startswith('#'):
                return line[:100] + ('...' if len(line) > 100 else '')

        return "Extracted Content"
    except:
        return "Extracted Content"


def _extract_publication_date_from_html(url: str) -> Optional[str]:
    """Extract publication date directly from webpage HTML"""
    try:
        # Fetch HTML content
        proxies = {
            'http': config.PROXY_CACHE,
            'https': config.PROXY_CACHE,
        }

        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, proxies=proxies)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Common meta tags for publication date
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[property="article:modified_time"]',
            'meta[name="date"]',
            'meta[name="pubdate"]',
            'meta[name="published"]',
            'meta[name="datePublished"]',
            'meta[name="publication-date"]',
            'meta[property="og:published_time"]',
            'meta[name="DC.date"]',
            'meta[name="DC.date.created"]',
            'meta[itemprop="datePublished"]',
            'meta[itemprop="dateModified"]'
        ]

        for selector in meta_selectors:
            meta_tag = soup.select_one(selector)
            if meta_tag:
                content = meta_tag.get('content') or meta_tag.get('datetime')
                if content:
                    try:
                        # Parse and standardize the date
                        parsed_date = dateutil.parser.parse(content)
                        return parsed_date.isoformat()
                    except:
                        continue

        # Check for time tags with datetime attribute
        time_tags = soup.find_all('time', {'datetime': True})
        for time_tag in time_tags:
            try:
                parsed_date = dateutil.parser.parse(time_tag['datetime'])
                return parsed_date.isoformat()
            except:
                continue

        # JSON-LD structured data
        json_ld_scripts = soup.find_all(
            'script', {'type': 'application/ld+json'})
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    date_fields = ['datePublished',
                                   'dateCreated', 'dateModified']
                    for field in date_fields:
                        if field in data:
                            parsed_date = dateutil.parser.parse(data[field])
                            return parsed_date.isoformat()
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for field in date_fields:
                                if field in item:
                                    parsed_date = dateutil.parser.parse(
                                        item[field])
                                    return parsed_date.isoformat()
            except:
                continue

        return None

    except Exception as e:
        logger.debug(f"Error extracting publication date from {url}: {e}")
        return None


def _jina_reader(url: str, max_tokens: int) -> MCPToolResult:
    """Get content using Jina Reader with enhanced error handling"""
    try:
        headers = {
            'Authorization': f'Bearer {config.URL_CRAWLER_API_KEYS}',
            'Content-Type': 'application/json'
        }

        jina_url = f"https://r.jina.ai/{url}"
        proxies = {
            'http': config.PROXY_CACHE,
            'https': config.PROXY_CACHE,
        }

        response = requests.get(
            jina_url, headers=headers, proxies=proxies, timeout=30)

        # Enhanced error handling for 402 and other HTTP errors
        if response.status_code == 402:
            error_msg = (
                f"Jina Reader API returned 402 Payment Required. "
                f"This typically means: "
                f"1) API key is invalid or expired, "
                f"2) Account has insufficient credits/tokens, "
                f"3) API key needs activation. "
                f"Please check your account at https://jina.ai/reader/ "
                f"Response: {response.text[:200]}"
            )
            logger.error(f"Jina Reader 402 error for URL {url}: {error_msg}")
            return MCPToolResult(success=False, error=error_msg)

        elif response.status_code == 429:
            error_msg = (
                f"Jina Reader API rate limit exceeded (429). "
                f"Please wait before making more requests. "
                f"Response: {response.text[:200]}"
            )
            logger.warning(
                f"Jina Reader rate limit for URL {url}: {error_msg}")
            return MCPToolResult(success=False, error=error_msg)

        elif response.status_code == 401:
            error_msg = (
                f"Jina Reader API returned 401 Unauthorized. "
                f"Please check your API key configuration. "
                f"Response: {response.text[:200]}"
            )
            logger.error(f"Jina Reader auth error for URL {url}: {error_msg}")
            return MCPToolResult(success=False, error=error_msg)

        # Raise for other HTTP errors
        response.raise_for_status()

        content = response.text

        # Log successful request
        logger.info(
            f"Jina Reader successful for URL {url}: {len(content)} characters extracted")

        # Truncate if needed
        if max_tokens and len(content.split()) > max_tokens:
            words = content.split()[:max_tokens]
            content = ' '.join(words) + '...'
            logger.info(f"Content truncated to {max_tokens} tokens")

        return MCPToolResult(success=True, data=content)

    except requests.exceptions.RequestException as e:
        error_msg = f"Jina Reader network error: {str(e)}"
        logger.error(f"Jina Reader network error for URL {url}: {e}")
        return MCPToolResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Jina Reader failed: {str(e)}"
        logger.error(f"Jina Reader unexpected error for URL {url}: {e}")
        return MCPToolResult(success=False, error=error_msg)


@tool_info_decorator
@tool(metadata={
    "description": "Jina reader tools",
    "cacheability": "TRANSIENT",
    "expiration_time": config.EXPIRATION_TIME,
    "primary_args": ["url", "session_id"]
})
def jina_reader(
    url: Union[str, List[str]],
    session_id: str,
    output_file: Union[str, List[str], None] = None,
    max_tokens: int = 8000,
    target_directory: str = "extracted_content",
    overwrite: bool = False,
    include_metadata: bool = True,
    max_workers: int = 5
) -> Any:
    """
    Extract markdown content from URL(s) using Jina Reader API and save to file(s)

    Args:
        url: Single URL (string) or multiple URLs (list) to extract content from
        session_id: The conversation ID
        output_file: Single filename or list of filenames (auto-generated if not provided)
        max_tokens: Maximum tokens to extract from each URL
        target_directory: Directory to save the file(s) (relative to workspace)
        overwrite: Whether to overwrite existing files
        include_metadata: Whether to include extraction metadata in the file headers
        max_workers: Maximum number of concurrent extraction requests (for multiple URLs)
    """
    try:
        # Normalize inputs to lists for unified processing
        urls = [url] if isinstance(url, str) else url
        output_files = None

        if output_file is not None:
            if isinstance(output_file, str):
                output_files = [output_file] if isinstance(
                    url, str) else [output_file] * len(urls)
            else:
                output_files = output_file
                if len(output_files) != len(urls):
                    raise ValueError(
                        f"Number of output_files ({len(output_files)}) must match number of URLs ({len(urls)})")
        # Validate URLs
        for u in urls:
            if not u or not u.startswith(('http://', 'https://')):
                raise ValueError(
                    f"Invalid URL provided: {u}. URL must start with http:// or https://")
        # Prepare target directory
        workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
        workspace_path.mkdir(exist_ok=True)
        save_directory = workspace_path / target_directory
        save_directory.mkdir(parents=True, exist_ok=True)

        def process_single_url(url_index: int) -> Dict[str, Any]:
            """Process a single URL and save to file"""
            current_url = urls[url_index]
            current_output_file = output_files[url_index] if output_files else None

            try:
                # Extract content using Jina Reader
                logger.info(f"Extracting content from URL: {current_url}")
                content_result = _jina_reader(current_url, max_tokens)

                if not content_result.success:
                    raise ValueError(
                        f"Failed to extract content: {content_result.error}")
                # Extract publication date for metadata
                publication_date = None

                # Generate filename if not provided
                if not current_output_file:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(current_url)
                    domain = parsed_url.netloc.replace('www.', '')
                    path_parts = [
                        part for part in parsed_url.path.split('/') if part]

                    if path_parts:
                        # Use last path component as base name
                        base_name = path_parts[-1]
                        # Remove file extension if present
                        if '.' in base_name:
                            base_name = base_name.rsplit('.', 1)[0]
                    else:
                        base_name = domain.replace('.', '_')

                    # Clean filename
                    base_name = re.sub(r'[^\w\-_\.]', '_', base_name)
                    base_name = re.sub(r'_+', '_', base_name).strip('_')

                    # Add timestamp to make unique
                    timestamp = int(time.time())
                    current_output_file = f"{base_name}_{timestamp}.md"

                # Ensure .md extension
                if not current_output_file.endswith('.md'):
                    current_output_file += '.md'

                file_path = save_directory / current_output_file

                # Check if file exists
                if file_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"File already exists: {file_path.relative_to(workspace_path)}. Use overwrite=True to replace it.")

                # Prepare content with metadata
                markdown_content = content_result.data

                if include_metadata:
                    # Add metadata header
                    metadata_header = f"""---
title: "{_extract_title_from_content(markdown_content)}"
source_url: {current_url}
extracted_at: {datetime.now().isoformat()}
publication_date: {publication_date if publication_date else "unknown"}
content_length: {len(markdown_content)}
word_count: {len(markdown_content.split())}
---

"""
                    markdown_content = metadata_header + markdown_content

                # Save to file
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(markdown_content, encoding='utf-8')
                    logger.info(
                        f"Successfully saved extracted content to: {file_path.relative_to(workspace_path)}")
                except Exception as e:
                    raise RuntimeError(f"Failed to save file: {e}")

                # Prepare successful result
                return {
                    'url': current_url,
                    'success': True,
                    'output_file': str(file_path.relative_to(workspace_path)),
                    'file_size': file_path.stat().st_size,
                    'content_length': len(content_result.data),
                    'word_count': len(content_result.data.split()),
                    'publication_date': publication_date,
                    'extracted_at': datetime.now().isoformat(),
                    'metadata_included': include_metadata
                }

            except Exception as e:
                logger.error(f"Error processing URL {current_url}: {e}")
                raise e

        # Process URLs (single or concurrent)
        if len(urls) == 1:
            # Single URL - process directly
            results = [process_single_url(0)]
        else:
            # Multiple URLs - process concurrently
            results = []
            with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as executor:
                # Submit all extraction tasks
                future_to_index = {executor.submit(
                    process_single_url, i): i for i in range(len(urls))}

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        index = future_to_index[future]
                        current_url = urls[index]
                        logger.error(
                            f"Error processing extraction for '{current_url}': {e}")
                        results.append({
                            'url': current_url,
                            'success': False,
                            'error': str(e),
                            'output_file': None,
                            'file_size': 0
                        })
            # Sort results to maintain original URL order
            url_order = {urls[i]: i for i in range(len(urls))}
            results.sort(key=lambda x: url_order.get(x['url'], float('inf')))

        # Calculate summary stats
        successful_extractions = len(
            [r for r in results if r.get('success', False)])
        total_files_saved = len([r for r in results if r.get(
            'success', False) and r.get('output_file')])
        total_size = sum(r.get('file_size', 0)
                         for r in results if r.get('success', False))

        # Generate status message following context management philosophy
        if isinstance(url, str):
            status_msg = f"Content extraction completed successfully. Extracted {max_tokens} tokens from 1 URL and saved to {save_directory.relative_to(workspace_path)}. Use file reading tools to examine the extracted content."
        else:
            status_msg = f"Batch content extraction completed successfully. Processed {len(urls)} URLs with {successful_extractions} successful extractions and {total_files_saved} files saved to {save_directory.relative_to(workspace_path)}. Use file reading tools to examine the extracted content."

        return MCPToolResult(
            success=True,
            data=status_msg,
            metadata={
                'extraction_method': 'jina_reader',
                'save_directory': str(save_directory.relative_to(workspace_path)),
                'total_urls': len(urls),
                'successful_extractions': successful_extractions,
                'failed_extractions': len(urls) - successful_extractions,
                'total_files_saved': total_files_saved,
                'total_size_bytes': total_size,
                'concurrent_workers': min(max_workers, len(urls)) if len(urls) > 1 else 1
            }
        )

    except Exception as e:
        logger.error(f"Jina Reader extraction and save failed: {e}")
        raise e
