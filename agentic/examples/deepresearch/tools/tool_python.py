import re
from typing import Dict, List, Optional, Union, Any
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from requests.exceptions import Timeout
import random
import time

from examples.deepresearch.config.deepresearch_config import SANDBOX_FUSION_ENDPOINTS


CHINESE_CHAR_RE = re.compile(r'[一-鿿]')


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n\n'

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'
            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def observation(self, tool, tool_dict, tool_results, empty_mode=False, readpage=False, max_observation_length=None, tokenizer=None):
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results

    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def call(self, params, files=None, timeout=50, **kwargs) -> str:
        try:
            from sandbox_fusion import run_code, RunCodeRequest, RunStatus
            code = params
            last_error = None
            for attempt in range(8):
                try:
                    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
                    print(f"Attempt {attempt + 1}/5 using endpoint: {endpoint}")

                    code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), max_attempts=1, client_timeout=timeout, endpoint=endpoint)
                    print("[Python] Code Result", code_result)
                    result = []
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")
                    if code_result.run_result.execution_time >= timeout-1:
                        result.append(f"[PythonInterpreter Error] TimeoutError: Execution timed out.")
                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')
                    return result if result.strip() else 'Finished execution.'

                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:
                        return last_error
                    continue

                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"