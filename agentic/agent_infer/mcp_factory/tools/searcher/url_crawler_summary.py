from ..tool_register import tool_info_decorator, base_client
from ..tool_result_format import MCPToolResult
from typing import Dict, Any, Union, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import time
from bs4 import BeautifulSoup
import dateutil.parser
import logging
from ..filer.file_write import file_write
import random
import requests
import os
import config.config as config

logger = logging.getLogger(__name__)


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
        }, proxies=proxies, verify=False)
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
                    except ValueError:
                        continue

        # Check for time tags with datetime attribute
        time_tags = soup.find_all('time', {'datetime': True})
        for time_tag in time_tags:
            try:
                parsed_date = dateutil.parser.parse(time_tag['datetime'])
                return parsed_date.isoformat()
            except ValueError:
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
            except ValueError:
                continue

        return None

    except Exception as e:
        logger.debug(f"Error extracting publication date from {url}: {e}")
        return None


def _summarize(
    goal: str,
    webpage_content: str,
) -> str:
    # Ensure the input is a valid, non-empty string.
    user_prompt = f"""Please process the following webpage content and user goal to extract relevant information:

                    ## **Webpage Content** 
                    {webpage_content}

                    ## **User Goal**
                    {goal}

                    ## **Task Guidelines**
                    1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
                    2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
                    3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

                    **Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
                    """
    try:
        response = base_client.chat.completions.create(
            model=config.DEFAULT_MODEL_NAME,  # Using a more specific config variable name
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        summary = response.choices[0].message.content
        logging.info("Successfully summarized the text.")
        return summary.strip() if summary else webpage_content
    except Exception as e:
        logging.error(f"An error occurred during summarization: {e}")
        # Fallback to returning the original text in case of an error.
        return webpage_content


def _content_extractor(goal: str, url: str, max_tokens: int) -> MCPToolResult:
    """Get content using URL content extractor"""
    max_retry_num = 5
    sleep_time = 5
    retry_num = 0
    while True:
        retry_num += 1
        try:
            headers = {
                "Authorization": f"Bearer {config.URL_CRAWLER_API_KEYS}",
                'Content-Type': 'application/json'
            }
            proxies = {
                'http': config.PROXY_CACHE,
                'https': config.PROXY_CACHE,
            }
            jina_url = f"https://r.jina.ai/{url}"

            response = requests.get(
                jina_url, headers=headers, timeout=30, proxies=proxies, verify=False)
            content = response.text
            summary_content = _summarize(goal, content)

            return MCPToolResult(success=True, data=summary_content)
            # Users need to implement this - placeholder for custom URL crawler

        except Exception as e:
            if retry_num == max_retry_num:
                return MCPToolResult(success=False, error=f"Content extractor failed: {e}")
            else:
                time.sleep(sleep_time)


@tool_info_decorator
def url_crawler(
        goal: str,
        documents: List[Dict],
        session_id: str,
        max_tokens_per_url: int = 100000,
        include_metadata: bool = True,
        max_workers: int = 10
) -> Any:
    """
    Extract content from web pages using configurable URL crawler API and retrieve relevant information for goal. 
    Input is a list of documents with metadata including URL and local file path for saving extracted content.

    Args:
        goal: The goal of the visit for webpage(s).
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
        def process_single_document(doc: Dict) -> Dict[str, Any]:
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
                    goal, url, max_tokens_per_url)

                if not content_result.success:
                    result_base['error'] = content_result.error
                    return result_base

                content = content_result.data
                if not content:
                    result_base['error'] = "Extracted content is empty"
                    return result_base

                # Save content to file
                write_result = file_write(
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
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(documents))) as executor:
            # Submit all processing tasks
            future_to_doc = {executor.submit(
                process_single_document, doc): doc for doc in documents}

            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    url = doc['url']
                    logger.error(
                        f"Error processing extraction for '{url}': {e}")

                    # Try to extract publication date even if processing failed
                    publication_date = None

                    results.append({
                        'url': url,
                        'file_path': doc['file_path'],
                        'title': doc.get('title'),
                        'time': doc.get('time'),
                        'success': False,
                        'error': str(e),
                        'publication_date': publication_date,
                        'extraction_timestamp': time.time(),
                        'write_success': False
                    })

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
