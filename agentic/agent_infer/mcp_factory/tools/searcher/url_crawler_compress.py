import asyncio
import os
import httpx
from .url_crawler import _content_extractor
from ..tool_register import tool_info_decorator
from typing import Dict, Any, List, Optional
from ..tool_result_format import MCPToolResult
import time
import logging
from ..filer.file_write import file_write
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from config.config import NLTK_DATA_PATH
nltk.data.path.insert(0, NLTK_DATA_PATH)

logger = logging.getLogger(__name__)

async def call_emb(string_list):
    embedding_url = os.getenv("EMBEDDING_URL", "")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
    if not embedding_url:
        raise RuntimeError("EMBEDDING_URL must be configured before using url_crawler_compress")
    headers = {"Content-Type": "application/json"}
    if embedding_api_key:
        headers["Authorization"] = f"Bearer {embedding_api_key}"
    data = {
        "model": 'qwen3-embedding-4b',
        "input": string_list
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(embedding_url, headers=headers, json=data)
        if response.status_code == 200:
            vectors = []
            for i in range(len(string_list)):
                vectors.append(response.json()["data"][i]["embedding"])
            return vectors

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 30, language: str = None):
    """
        text (str): 做切分的文本。
        chunk_size (int): 切分块的大小，以字符为单位。
        chunk_overlap (int): 重叠块的大小。
        language (str):输入语言是中文还是英文。
    """

    def remove_urls(text):
        text_lines = text.split("\n")
        url_pattern = re.compile(r'https?://\S+|ftp://\S+|www\.\S+')
        text_lines_new = [url_pattern.sub('', x) for x in text_lines]
        return "\n".join(text_lines_new)

    text = remove_urls(text)

    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

    def detect_language_optimized(text, chinese_ratio_threshold=0.5):
        """
        使用正则表达式，以更快的方式判断字符串的语言。
        """
        if not isinstance(text, str) or not text:
            raise ValueError('text is not a str')

        chinese_chars = chinese_pattern.findall(text)
        english_chars = english_pattern.findall(text)

        chinese_count = len(chinese_chars)
        english_count = len(english_chars)

        total_effective_count = chinese_count + english_count

        if total_effective_count == 0:
            return 'english'

        chinese_ratio = chinese_count / total_effective_count

        if chinese_ratio > chinese_ratio_threshold:
            return 'chinese'
        else:
            return 'english'

    if language is None:
        language = detect_language_optimized(text, 0.1)
        print(f"该文档判定为{language}文档")
        # chunk_size = 200
        # chunk_overlap = 20
        # if language=='chinese':
        #     chunk_size = 100
        #     chunk_overlap = 10
    if language.lower() == 'chinese':
        sentences = re.split(r'([。！？]|\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        sentences = sent_tokenize(text)
        # pass
    # print("切分")
    ans_chunks = []
    current_chunk = []
    current_len = 0
    # print('sentence', sentences)
    max_chunk = int(1.2 * chunk_size)
    for sent in sentences:
        sent_len = len(sent)

        # if the sentence if too long.
        if sent_len > max_chunk:
            for i in range(0, len(sent), max_chunk):
                ans_chunks.append(sent[i:i + max_chunk])
            last_one = ans_chunks.pop()
            current_chunk = [last_one]
            current_len = sum(len(s) for s in current_chunk)
            continue

        if current_len + sent_len > chunk_size and current_chunk:
            # 保存当前块并保留重叠部分
            ans_chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[int(-max(1, len(current_chunk) * chunk_overlap // chunk_size)):]
            current_len = sum(len(s) for s in current_chunk)

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        ans_chunks.append(' '.join(current_chunk))
    return ans_chunks

## Compress content
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

async def compress_content(content):
    max_chunk_num = 100
    chunk_num = 10000
    chunk_size = 100
    while chunk_num > max_chunk_num:
        final_result = chunk_text(content, chunk_size, chunk_size / 10)
        chunk_num = len(final_result)
        chunk_size += 50

    Emb = await call_emb(final_result)
    n = len(Emb)

    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity_matrix[i, j] = cosine_similarity(Emb[i], Emb[j])

    similar_items = []
    similarity_threshold = 0.7
    text_span_threshold = round(n / 2)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] * np.exp(-(j - i - 1) / (
                    -text_span_threshold / np.log(similarity_threshold))) > similarity_threshold:
                similar_items.append([i, j])

    deleted_items = []
    for i in range(len(similar_items)):
        if similar_items[i][0] not in deleted_items:
            deleted_items.append(similar_items[i][1])
    deleted_items = set(deleted_items)

    cleaned_result = [x for i, x in enumerate(final_result) if i not in deleted_items]

    joint_result = ''
    for i in range(len(cleaned_result)):
        joint_result += cleaned_result[i]

    print(f'文档已压缩. 压缩前文档字符数: {len(content)}; 压缩后文档字符数: {len(joint_result)}.')
    return joint_result

@tool_info_decorator
async def url_crawler_compress(
        documents: List[Dict],
        session_id: str,
        max_tokens_per_url: int = 100000,
        include_metadata: bool = True,
        max_workers: int = 10,
) -> Any:
    """
        Extract LLM-friendly content from URLs using configurable crawler service.
        Content is saved to specified file paths.

        Users need to implement their own URL crawler. The return format should include:
        - Extracted text content from the URL
        - Metadata like title, publication date, word count
        - Success/error status for each URL

        Args:
            documents: List of document dictionaries containing:
                - url: Web page URL to extract
                - file_path: Local path to save extracted content
                - title: (Optional) Web page title
                - time: (Optional) Web page publication time
            session_id: The conversation ID
            max_tokens_per_url: Maximum tokens per URL result
            include_metadata: Whether to include metadata about extraction
            max_workers: Maximum number of concurrent extraction requests
        """
    try:
        async def process_single_document(doc: Dict) -> Dict[str, Any]:
            """Process a single document: extract content and save to file"""
            url = doc['url']
            file_path = doc['file_path']
            title = doc.get('title')
            doc_time = doc.get('time')

            result_base = {
                'url': url,
                'file_path': file_path,
                'title': title,
                'time': doc_time,
                'success': False,
                'error': None,
                'content_length': 0,
                'word_count': 0,
                'publication_date': None,
                'extraction_timestamp': time.time(),
                'write_success': False
            }

            try:
                # Extract publication date from the webpage
                publication_date = None
                result_base["publication_date"] = publication_date

                # Extract content using content extractor
                content_result = _content_extractor(
                    url, max_tokens_per_url)

                if not content_result.success:
                    result_base['error'] = content_result.error
                    return result_base

                content = content_result.data
                if not content:
                    result_base['error'] = "Extracted content is empty"
                    return result_base

                content = await compress_content(content)

                # Save content to file
                write_result = await file_write(
                    file_path=file_path,
                    content=content,
                    session_id=session_id,
                    create_dirs=True
                )

                # Build success result
                result = {
                    **result_base,
                    'success': True,
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'publication_date': publication_date,
                    'write_success': True
                }

                if include_metadata:
                    result['metadata'] = {
                        'truncated': len(content.split()) >= max_tokens_per_url,
                        'has_publication_date': publication_date is not None,
                        'date_extraction_method': 'html_parsing' if publication_date else None,
                        'file_size': len(content.encode('utf-8'))
                    }

                return result

            except Exception as e:
                logger.error(f"Error processing document {url}: {e}")
                # Try to extract publication date even if processing failed
                publication_date = None

                return {
                    **result_base,
                    'error': str(e),
                    'publication_date': publication_date
                }

        # Execute processing concurrently
        tasks = [
            process_single_document(doc)
            for doc in documents
        ]
        oral_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, res in enumerate(oral_results):
            doc = documents[i]
            if isinstance(res, Exception):
                logger.error(f"Unhandled exception for {doc['url']}: {res}")
                results.append({
                    'url': doc['url'],
                    'file_path': doc['file_path'],
                    'title': doc.get('title'),
                    'time': doc.get('time'),
                    'success': False,
                    'error': str(res),
                    'publication_date': None,
                    'extraction_timestamp': time.time(),
                    'write_success': False
                })
            else:
                results.append(res)

        # Sort results to maintain original document order
        url_order = {doc['url']: i for i, doc in enumerate(documents)}
        results.sort(key=lambda x: url_order.get(x['url'], float('inf')))

        successful_extractions = len(
            [r for r in results if r.get('success', False)])
        successful_writes = len(
            [r for r in results if r.get('write_success', False)])

        return {"success": True,
                "data": results,
                "metadata": {
                    'total_documents': len(documents),
                    'successful_extractions': successful_extractions,
                    'successful_writes': successful_writes,
                    'failed_processing': len(documents) - successful_extractions,
                    'concurrent_workers': min(max_workers, len(documents))
                }
                }

    except Exception as e:
        logger.error(f"URL crawler batch processing failed: {e}")
        return MCPToolResult(success=False, error=str(e))
