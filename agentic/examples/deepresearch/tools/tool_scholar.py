import json
import base64
import http.client
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor

from qwen_agent.tools.base import BaseTool, register_tool

from examples.deepresearch.config.deepresearch_config import SERPER_KEY, PROXY_HOST, PROXY_PORT, PROXY_USER, PROXY_PASS


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            },
        },
        "required": ["query"],
    }

    def google_scholar_with_serp(self, query: str):
        SERPER_HOST = "google.serper.dev"

        if PROXY_HOST and PROXY_PORT:
            conn = http.client.HTTPSConnection(PROXY_HOST, int(PROXY_PORT))
            auth = base64.b64encode(f"{PROXY_USER}:{PROXY_PASS}".encode()).decode()
            conn.set_tunnel(SERPER_HOST, 443, headers={"Proxy-Authorization": f"Basic {auth}"})
        else:
            conn = http.client.HTTPSConnection(SERPER_HOST)

        payload = json.dumps({
            "q": query,
        })
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }
        for i in range(5):
            try:
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google Scholar Timeout, return None, Please try again later."
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))
        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    link_info = "no available link"
                    if "pdfUrl" in page:
                        link_info = "pdfUrl: " + page["pdfUrl"]

                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])

                    redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self.google_scholar_with_serp(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.google_scholar_with_serp, query))
            response = "\n=======\n".join(response)
        return response