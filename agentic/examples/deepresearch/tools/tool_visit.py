import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI
import time
import tiktoken

from examples.deepresearch.config.deepresearch_config import (
    JINA_API_KEYS, SUMMARY_API_KEY, SUMMARY_API_BASE, SUMMARY_MODEL_NAME,
    VISIT_SERVER_TIMEOUT, WEBCONTENT_MAXLENGTH, VISIT_SERVER_MAX_RETRIES,
)
from examples.deepresearch.tools.prompt import EXTRACTOR_PROMPT


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            response = self.readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url:
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)

        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def call_server(self, msgs, max_retries=2):
        client = OpenAI(
            api_key=SUMMARY_API_KEY,
            base_url=SUMMARY_API_BASE,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=SUMMARY_MODEL_NAME,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content
            except Exception as e:
                if attempt == (max_retries - 1):
                    return ""
                continue

    def jina_readpage(self, url: str) -> str:
        max_retries = 3
        timeout = 50

        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {JINA_API_KEYS}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"
            print(service)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        summary_page_func = self.call_server

        content = self.html_readpage_jina(url)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw = summary_page_func(messages, max_retries=VISIT_SERVER_MAX_RETRIES)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=VISIT_SERVER_MAX_RETRIES)
                summary_retries -= 1

            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=VISIT_SERVER_MAX_RETRIES)
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"

            return useful_information

        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information