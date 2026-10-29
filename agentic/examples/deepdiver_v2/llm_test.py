import os

import litellm

if __name__ == '__main__':

    response = litellm.completion(
        model="qwen3-32b",
        messages=[{"role": "user", "content": "你好"}],
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.environ["OPENAI_BASE_URL"]
    )
    print(response)