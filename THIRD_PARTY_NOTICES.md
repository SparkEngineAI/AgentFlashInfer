# 第三方来源与参考项目

以下代码包含对上游实现的集成或修改。保留各文件已有的版权和许可证声明；Apache-2.0 全文见 [LICENSE](LICENSE)。

| 组件 | 本项目位置与修改 | 上游 | 许可证 |
| --- | --- | --- | --- |
| DeepResearch | [Agent 服务](agentic/examples/deepresearch/README.md)：增加服务接口、SSE 推送、模型路由、提前停止和上下文压缩 | [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Apache-2.0 |
| vLLM | [配置与调度 patch](agentic/inference/pangu_infer/patches/vllm/README.md)、[后缀解码适配](agentic/inference/pangu_infer/pangu/accelerators/speculative_decoding/suffix.py)、[负载均衡代理示例](agentic/inference/serving/disaggregate_prefill/load_balance_proxy_server_example.py) | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| vLLM-Ascend / vLLM | [旋转位置编码适配](agentic/inference/pangu_infer/models/vllm_ascend/rotary_embedding.py)，保留 Huawei 和 vLLM 版权声明 | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| GPT-NeoX / Hugging Face Transformers | [模型实现](agentic/inference/pangu_infer/models/vllm_ascend/modeling_openpangu_embedded.py)：适配模型结构，保留 Huawei、EleutherAI 和 HuggingFace 版权声明 | [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)、[huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| verl | [序列长度均衡工具](agentic/agent_infer/unified_store/utils/seqlen_balancing.py)，保留 Huawei 和 ByteDance 版权声明 | [volcengine/verl](https://github.com/volcengine/verl) | Apache-2.0 |

## 参考项目

[Fusion Mem](agentic/features/fusion_mem/README.md) 的参考项目：[mem0](https://github.com/mem0ai/mem0)（Apache-2.0）。
