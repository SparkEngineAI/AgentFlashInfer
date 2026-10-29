import os
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from agent_infer.mcp_factory.tools.tool_register import tool_info_decorator, base_client

from agent_infer.mcp_factory.tools.seeker.semantic_store import get_files, TextChunker, read_txt_file
import asyncio
import aiofiles
from fusion_mem.client import AsyncMemoryClient
import config.config as config
MEMORY_BASE_URL = config.MEMORY_BASE_URL
client = AsyncMemoryClient(MEMORY_BASE_URL)
@tool_info_decorator
async def semantic_search(queries: List[str],session_id: str, max_results_per_query: int = 15):
        """
        Search all files stored in the local workspace with regard to the semantic meaning of the query.
        Return relevant chunks of information
        Args:
            queries: List of strings to be looked at
            max_results_per_query: Maximum number of returned text chunks for each query
        """

        workspace_path = Path(os.path.join(config.WORKSPACE_PATH)) / session_id
        hot_file_record = Path(workspace_path) / "hot_file.json"
        text_chunker = TextChunker()

        if hot_file_record.exists():
            try:
                async with aiofiles.open(str(hot_file_record), 'r', encoding='utf-8') as f:
                    content = await f.read()
                    hot_record = json.loads(content)
                    hot_record = defaultdict(int, hot_record)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        else:
            hot_record = defaultdict(int)

        if not MEMORY_BASE_URL:
            print("Failed to add memory to fusion mem module.")
            return

        new_files = await get_files(workspace_path, workspace_path=workspace_path)
        tasks = []
        for _new_file in new_files:
            full_path = os.path.join(workspace_path, _new_file)
            content = await read_txt_file(full_path)
            if content:
                response = text_chunker.chunk_text(content)
                contents = [{"content": memory_contents} for memory_contents in response]
                task = client.add_scenario_memories(messages=contents, session_id=session_id)
                tasks.append(task)
        await asyncio.gather(*tasks)
        results = []
        for _query in queries:
            response = await client.search_scenario_memories(_query, session_id=session_id)
            ans_response = []
            num_recalls = []
            hot_response = []
            alpha =  1
            gamma = 1
            for _response in response["results"]:
                ans_response.append(_response['memory'])

                hot_record[str(_response['id'])] += 1 # 记录出现的次数。
                num_recalls.append(hot_record[str(_response['id'])])

                original_hot = alpha * hot_record[str(_response['id'])] # + gamma * np.exp(-int(_response['id']) / num_entities[0]['count(*)'])
                penalty = np.exp(_response['score']) - 1 # np.exp(_response['distance']) - 1
                hot_response.append(penalty * original_hot)

            # 保存heat_value
            async with aiofiles.open(str(hot_file_record), 'w', encoding='utf-8') as f:
                json_str  = json.dumps(hot_record,indent=2)
                await f.write(json_str)

            results.append({
                            "query": _query,
                            "success": True,
                            "heat_value": hot_response,
                            "num_recalls": num_recalls,
                            "results": ans_response,
                            "timestamp": time.time()
                            })
        if results:
            return results
        else:
            return ["There is no data in the local milvus_demo. First searching some information on the website."]
