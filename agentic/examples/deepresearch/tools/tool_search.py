import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional
import http.client
import base64

from qwen_agent.tools.base import BaseTool, register_tool

from examples.deepresearch.config.deepresearch_config import SERPER_KEY, PROXY_HOST, PROXY_PORT, PROXY_USER, PROXY_PASS


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('一' <= char <= '鿿' for char in text)

        SERPER_HOST = "google.serper.dev"

        if PROXY_HOST and PROXY_PORT:
            conn = http.client.HTTPSConnection(PROXY_HOST, int(PROXY_PORT))
            auth = base64.b64encode(f"{PROXY_USER}:{PROXY_PASS}".encode()).decode()
            conn.set_tunnel(SERPER_HOST, 443, headers={"Proxy-Authorization": f"Basic {auth}"})
        else:
            conn = http.client.HTTPSConnection(SERPER_HOST)

        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })

        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json',
        }

        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))
        print(f"result={results}")

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self.search_with_serp(query)
        else:
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)

        return response