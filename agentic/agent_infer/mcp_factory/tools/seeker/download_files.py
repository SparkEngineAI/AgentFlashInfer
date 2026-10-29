import asyncio

import aiofiles
import httpx

from ..tool_register import tool_info_decorator
import logging
from pathlib import Path
from typing import Any, List
import os
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
import config.config as config

load_dotenv()

logger = logging.getLogger(__name__)


@tool_info_decorator
async def download_files(
    urls: List[str],
    session_id: str,
    target_directory: str = None,
    overwrite: bool = False,
    max_file_size_mb: int = 100
) -> Any:
    """
    Download files from URLs

    Args:
        urls: List of URLs to download
        session_id: The conversation ID
        target_directory: Directory to save files (default: workspace)
        overwrite: Whether to overwrite existing files
        max_file_size_mb: Maximum file size in MB
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))
    if target_directory:
        download_dir = Path(target_directory)
    else:
        download_dir = workspace_path / "downloads"

    await asyncio.to_thread(lambda: download_dir.mkdir(parents=True, exist_ok=True))

    results = []

    for url in urls:

        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or 'downloaded_file'

        # Ensure filename has extension
        if '.' not in filename:
            filename += '.html'  # Default extension

        file_path = download_dir / filename

        # Check if file exists
        exists = await asyncio.to_thread(file_path.exists)
        if exists and not overwrite:
            results.append({
                'url': url,
                'success': False,
                'error': 'File already exists',
                'file_path': str(file_path)
            })
            continue

        # Download file
        proxies = {
            "http://": config.PROXY_CACHE,
            "https://": config.PROXY_CACHE,
        }

        # 配置客户端（注意：verify=False 对应 httpx 的 verify=False）
        async with httpx.AsyncClient(proxies=proxies, verify=False, timeout=30) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()  # 如果状态码不是 2xx，会抛出异常

                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_file_size_mb * 1024 * 1024:
                    results.append({
                        'url': url,
                        'success': False,
                        'error': f'File too large (>{max_file_size_mb}MB)',
                        'file_path': None
                    })
                    continue

                # Save file
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)

                results.append({
                    'url': url,
                    'success': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size
                })
    return results
