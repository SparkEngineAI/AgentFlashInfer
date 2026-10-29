import asyncio
import time
from typing import List, Dict
import requests

from agent_infer.mcp_factory.tool_cache.tool_cache_lib import tool
from agent_infer.agent_specs.deepdiver_v2.search_compress_assistant import SEARCH_COMPRESS_ASSISTANT_PROMPT
from ..tool_register import tool_info_decorator
from typing import Any
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..tool_result_format import MCPToolResult
from copy import deepcopy
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from agent_infer.agent_specs.base_agent import logger
from config import config

logger = logging.getLogger(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=10)

def _generic_search(query: str, max_results: int) -> MCPToolResult:
    try:
        url = config.SEARCH_URL
        payload = json.dumps({
            "q": query,
            "num": min(max_results, 10)  # Serper allows max 10 per request
        })
        headers = {
            'X-API-KEY': config.SEARCH_API_KEY,
            'Content-Type': 'application/json'
        }
        proxies = None
        if config.PROXY_CACHE:
            proxies = {
                'http': config.PROXY_CACHE,
                'https': config.PROXY_CACHE,
            }
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        with requests.Session() as session:
            # 将重试策略挂载到 http 和 https 协议上
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            if not config.PROXY_CACHE:
                session.trust_env = False

            # 4. 发起请求 (增加 timeout)
            # timeout=(5, 10) -> 连接超时 5s，读取超时 10s
            response = session.post(
                url,
                data=payload,
                headers=headers,
                proxies=proxies,
                verify=False,
                timeout=(5, 20)
            )
        # response = requests.post(
        #     url, data=payload, headers=headers, proxies=proxies, verify=False)
            response.raise_for_status()
            # Transform your API response to required format
            search_results = {
                "organic": [
                    {
                        "title": result["title"],
                        "link": result["link"],
                        "snippet": result["snippet"],
                        "date": result.get("date", "unknown")
                    }
                    for result in response.json().get("organic", [])
                ]
            }
        return MCPToolResult(success=True, data=search_results)
    except Exception as e:
        return MCPToolResult(success=False, error=f"Generic search failed: {e}")


@tool_info_decorator
@tool(metadata={
    "description": "Batch web search using configurable search provider with concurrent processing",
    "cacheability": "TRANSIENT",
    "expiration_time": config.EXPIRATION_TIME,
    "primary_args": ["queries"],
})
async def batch_web_search(
        queries: List[str],
        max_results_per_query: int = 15,
        max_workers: int = 5
) -> Any:
    """
    Batch web search using configurable search provider with concurrent processing.
    Args:
        queries: List of search queries
        max_results_per_query: Maximum search results per query
        max_workers: Maximum number of concurrent search requests
    """

    try:
        # Ensure we never return more than 15 results per query
        actual_max_results = min(max_results_per_query, 15)

        async def search_single_query(query: str) -> Dict[str, Any]:
            """Search a single query"""
            try:
                search_results = _generic_search(query, actual_max_results)

                if not search_results.success:
                    return {
                        'query': query,
                        'success': False,
                        'error': search_results.error,
                        'results': []
                    }
                # Process search results
                search_data = search_results.data
                search_data["organic"] = search_data["organic"][:actual_max_results]
                # search_data.pop("searchParameters")
                return {
                    'query': query,
                    'success': True,
                    'results': search_data,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.error(f"Error searching query '{query}': {e}")
                return {
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'results': []
                }

        # Execute searches concurrently
        tasks = [
            search_single_query(query)
            for query in queries
        ]
        oral_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, res in enumerate(oral_results):
            query = queries[i]
            if isinstance(res, Exception):
                logger.error(f"Error processing search for '{query}': {res}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(res),
                    'results': []
                })
            else:
                results.append(res)
        # Sort results to maintain original query order
        query_order = {query: i for i, query in enumerate(queries)}
        results.sort(key=lambda x: query_order.get(
            x['query'], float('inf')))

        successful_queries = 0
        result_items = 0
        for result in results:
            if not result.get("success"):
                continue
            payload = result.get("results", [])
            items = payload.get("organic", []) if isinstance(payload, dict) else payload
            if items:
                successful_queries += 1
                result_items += len(items)
        logger.info(
            f"[search] batch_web_search completed: "
            f"successful_queries={successful_queries}, result_items={result_items}"
        )

        if config.get_search_compress_config().get('search_compress_sync_switch', False):
            logger.info(
                f'### w/ Semantic Compress:  before compress: {str(json.dumps(results, ensure_ascii=False).__len__())}')
            results = filter_single_search_sync(results)
            logger.info(f'### w/ Semantic Compress:  after compress: {str(json.dumps(results, ensure_ascii=False).__len__())}')
        return results
    except Exception as e:
        logger.error(f"Batch web search failed: {e}")
        raise



def filter_single_search_sync( single_result: List) -> List:
    def filter_single_query(one_query: Dict) -> Dict[str, Any]:
        tic = time.time()
        query = one_query['query']
        original_one_query = deepcopy(one_query)
        results = one_query['results']
        results_sent = deepcopy(results)
        for i in range(len(results_sent)):
            results_sent[i]['id'] = i + 1
        prompt = SEARCH_COMPRESS_ASSISTANT_PROMPT.format(**{"query": query, "results_sent": results_sent})
        try:
            assistant_content = send_request(prompt)
        except Exception as e:
            logger.info(f"Search compress fail: {e}")
            assistant_content = ''
        try:
            assistant_content = json.loads(assistant_content.lstrip(' [unused16] [unused17] '))
        except Exception as e:
            logger.info(f'Error in filter_single_search() json.loads(): {assistant_content=}, {e=}')
            return one_query

        indices = [-1] if assistant_content['indices'] == -1 else [int(i) for i in
                                                                   assistant_content['indices'].split(',')]
        filtered_organic = []
        if not (len(indices) == 1 and indices[0] == -1):
            filtered_organic = [results[i - 1] for i in indices]
        one_query['results'] = filtered_organic
        one_query['compress_time'] = time.time() - tic
        return one_query
    filtered_data = []
    for one_query in single_result:
        one_query = filter_single_query(one_query)
        filtered_data.append(deepcopy(one_query))
    return filtered_data

def send_request(prompt) ->str:
    compress_config = config.get_search_compress_model_config()
    url = compress_config.get('url')
    api_key = compress_config.get('api_key', None)
    model_name = compress_config.get('model_name', None)
    temperature = compress_config.get('temperature', None)
    max_tokens = compress_config.get('max_tokens', None)
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers.update({"Authorization": api_key})
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        assistant_content = response_json['choices'][0]['message']['content']
        return assistant_content
    except Exception as e:
        logger.info(f"Search compress fail: {e}")
        return ''
