from typing import Any, Sequence, Mapping

from agents import GenerationSpanData, Trace
from agents.tracing import get_trace_provider
from agents.tracing.spans import SpanImpl, TSpanData, Span, NoOpSpan

from examples.deepdiver_v2.tracing.processor import SSEProfessor

from contextlib import contextmanager
@contextmanager
def streaming_generation_span():
    """
    不再创建新 Span，而是获取当前活跃的 Span 并注入 push_chunk 方法。
    """
    provider = get_trace_provider()
    # 1. 直接拿到正在运行的那个 Generation Span (即 parent)
    current_span = provider.get_current_span()

    if not current_span:
        # 如果当前没在 span 里，返回一个 NoOp 避免报错
        yield NoOpSpan(None)
        return

    # 2. 动态给这个已存在的 Span 实例挂载 push_chunk 方法
    # 这样你就不用管 ID 了，它用的就是当前卡片的 ID
    def push_chunk(chunk_type: str, delta: Any) -> None:
        # 直接利用 current_span 自身的 ID 发送数据
        if hasattr(SSEProfessor, "on_span_stream"):
            SSEProfessor.on_span_stream(current_span, chunk_type, delta)

    # 将方法绑定到实例上
    current_span.push_chunk = push_chunk

    try:
        # 3. 返回这个被增强过的原有 Span
        yield current_span
    finally:
        # 清理工作（可选）
        if hasattr(current_span, 'push_chunk'):
            delattr(current_span, 'push_chunk')

