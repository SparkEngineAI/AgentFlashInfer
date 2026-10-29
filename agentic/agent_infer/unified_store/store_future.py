from typing import Optional, Union

import grpc

# 目前只支持 grpc.Future，将来易于扩展
FutureType = Union[grpc.Future]


class StoreFuture:
    """
    UnifiedStore 统一的Future类, 调用future.result()获得结果
    """

    def __init__(self, grpc_future: FutureType, extractor):
        self._f = grpc_future
        self._extractor = extractor

    def result(self, timeout: Optional[float] = None):
        """
        阻塞等待底层 Future 完成，然后把原生 response 交给 extractor，
        返回处理后的业务数据。
        """
        raw = self._f.result(timeout=timeout)
        return self._extractor(raw)
