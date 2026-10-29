from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import logging
logger = logging.getLogger(__name__)
# 服务基础URL（需与FastAPI服务地址一致，默认是http://localhost:8000）
BASE_URL = "http://0.0.0.0:8289"


def start_agent(
    user_input: str,
    agent_name: str = "assistant",
    stream: bool = False
) -> None:
    """
    调用POST /agents 启动代理并获取结果
    :param user_input: 用户输入（代理需要处理的任务）
    :param agent_name: 要使用的代理名称（默认是assistant）
    :param stream: 是否启用流式返回（默认False）
    """
    url = f"{BASE_URL}/agents"
    # 构造请求参数（与FastAPI的AgentRequest模型对应）
    payload = {
        "user_input": user_input,
        "agent_name": agent_name,
        "stream": stream
    }
    # 请求头（指定JSON格式）
    headers = {"Content-Type": "application/json"}

    try:
        if not stream:
            # 非流式响应：等待代理处理完成后一次性获取结果
            logger.info(f"\n=== 启动代理（{agent_name}）- 非流式 ===")
            logger.info(f"用户输入：{user_input}\n")
            logger.info("正在处理...（请稍候）")

            response = requests.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=None  # 非流式处理超时时间（根据任务复杂度调整）
            )
            response.raise_for_status()
            result = response.json()

            # 展示非流式结果
            logger.info("\n=== 代理处理结果 ===")
            logger.info(f"代理ID：{result['task_id']}")
            logger.info(f"处理状态：{'已完成' if result['completed'] else '处理中'}")
            logger.info(f"结果内容：\n{result['content']}")
            logger.info(f"时间戳：{result['timestamp']:.2f}（秒级）")

        else:
            # 流式响应：实时获取代理的处理过程（SSE格式）
            logger.info(f"\n=== 启动代理（{agent_name}）- 流式返回 ===")
            logger.info(f"用户输入：{user_input}\n")
            logger.info("正在流式接收结果...（按Ctrl+C可终止）\n")

            # 发送流式请求（stream=True）
            with requests.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                stream=True,
                timeout=None  # 流式处理不设超时（需手动终止）
            ) as response:
                response.raise_for_status()
                # 迭代解析流式数据（SSE格式：data: 内容\n\n）
                for line in response.iter_lines(decode_unicode=True):
                    if line:  # 跳过空行
                        # 去掉SSE的"数据前缀"，提取实际内容
                        if line.startswith("data: "):
                            content = line[len("data: "):]
                            if content == "[DONE]":
                                logger.info("\n=== 流式传输完成 ===")
                                break
                            logger.info(f"流式内容：{content}")

    except requests.exceptions.RequestException as e:
        logger.info(f"\n启动代理失败：{str(e)}")
        logger.info("可能原因：1.服务未启动 2.代理名称不存在 3.网络问题")


if __name__ == "__main__":
    # 交互式选择操作
    logger.info("=== FastAPI Agent服务客户端 ===")
    input_list = ['必须使用工具，《百年孤独》里最能活的人是谁？',
                  '调用网络搜索，陶渊明生平',
                  '在某个著名忍者团队中，存在一个具有以下特征的成员：1.在团队中担任主要攻击角色，2.拥有机械改造能力并擅长远程武器攻击，3.曾因特殊能力被复活后再次被击败。请问这个角色是谁？',
                  '该地铁线路是中国首条穿越某重要江河的轨道交通线路，位于武汉市，其一期工程北起某国家级经济技术开发区，东北起于东西湖区。该线路开通初期曾因施工中发生事故，导致交通中断两个多月。这条轨道交通线路的名称是什么？']
    all_results = []
    with ThreadPoolExecutor(max_workers=min(4, len(input_list))) as executor:
        # Submit all search tasks
        future_to_query = {executor.submit(
            start_agent, query): query for query in input_list}
        for future in as_completed(future_to_query):
            result = future.result()
            all_results.append(result)
