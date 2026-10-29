from typing import Any, List, Dict, Union

import ray
from ray._raylet import ObjectRef
from torch import Tensor


class GetExperienceHandle:
    """
    异步 get_experience 的句柄，内部封装了 actor.get_experience.remote(...) 返回的 ObjectRef。
    """

    def __init__(self, ref: ObjectRef):
        self.ref = ref


class MessageQueueClient:
    """
    注意: 目前实现仅是包装GRPOTransferDOck供前期对齐, 非最终版。

    MessageQueueClient 接受一个已经创建好的 Ray Actor handle,
    并通过一个额外的同步标志参数控制调用方式：

        mq.some_method(sync, *args, **kwargs)

    当 sync=True 时，执行 ray.get() 同步阻塞直到返回结果；
    当 sync=False 时，直接返回 ObjectRef, 不阻塞。
    其他参数就是TD对应函数的调用参数。

    另外，调用 mq.get(ref) 就相当于 ray.get(ref)。
    """

    def __init__(self, actor_handle):
        """
        :param actor_handle: 通过 ray.remote(RayGRPOTrainer).remote(config) 得到的 Actor 实例
        """
        self._actor = actor_handle

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")

        def _method(sync: bool, *args, **kwargs):
            fn = getattr(self._actor, name)

            # 1) put_experience: 自动 ray.put data_dict
            if name == "put_experience":
                data_dict: Dict[str, Union[Tensor, List[Tensor]]] = args[0]
                args = (ray.put(data_dict),) + args[1:]

            # 2) 发起远程调用
            obj_ref: ObjectRef = fn.remote(*args, **kwargs)

            # 3) get_experience 特殊处理
            if name == "get_experience":
                if sync:
                    # 同步模式：先拿 (batch_ref, idxs)，再拿 batch，直接返回
                    batch_ref, idxs = ray.get(obj_ref)
                    return ray.get(batch_ref), idxs
                else:
                    # 异步模式：返回一个专用的 Handle
                    return GetExperienceHandle(obj_ref)

            # 4) 其它方法：按 sync 决定是否 ray.get
            return ray.get(obj_ref) if sync else obj_ref

        return _method

    def get(self, handle_or_ref: Union[ObjectRef, GetExperienceHandle, Any]):
        """
        获取sync==false时执行的实际结果。
        """
        # 1) 如果是 GetExperienceHandle，先拿到 (batch_ref, idxs)
        if isinstance(handle_or_ref, GetExperienceHandle):
            batch_ref, idxs = ray.get(handle_or_ref.ref)
            batch = ray.get(batch_ref)
            return batch, idxs

        # 2) ObjectRef：直接 ray.get
        if isinstance(handle_or_ref, ObjectRef):
            return ray.get(handle_or_ref)

        # 3) 其它类型，直接返回
        return handle_or_ref
