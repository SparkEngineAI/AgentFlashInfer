import pandas as pd
import threading
import uuid
from typing import List, Dict, Any, Optional, Union


class DramCacheManager:
    """
       升级版数据管理类：支持自动 Data ID 生成与基于 ID 的管理。
       """

    def __init__(self):
        self._df = pd.DataFrame()
        self._lock = threading.Lock()
        # 定义主键列名，方便后续维护
        self.ID_COL = 'data_id'

    def _generate_id(self) -> str:
        """生成唯一的 data_id (UUID4 hex)"""
        return uuid.uuid4().hex

    def add_data(self, data: Dict[str, Any]) -> str:
        """
        添加单条数据。
        如果 data 中包含 'data_id' 则使用，否则自动生成。

        Returns:
            str: 该条数据的 data_id
        """
        # 1. 预处理：在加锁前生成 ID (减少锁持有时间)
        row_id = data.get(self.ID_COL)
        if not row_id:
            row_id = self._generate_id()
            data[self.ID_COL] = row_id

        with self._lock:
            new_row = pd.DataFrame([data])  # 列表包裹字典直接转 DataFrame

            if self._df.empty:
                self._df = new_row
            else:
                self._df = pd.concat([self._df, new_row], ignore_index=True)

        return row_id

    def add_batch(self, data_list: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加数据并返回对应的 ID 列表。
        """
        ids = []
        # 预处理数据，填充 ID
        for item in data_list:
            if self.ID_COL not in item:
                item[self.ID_COL] = self._generate_id()
            ids.append(item[self.ID_COL])

        with self._lock:
            new_rows = pd.DataFrame(data_list)
            if self._df.empty:
                self._df = new_rows
            else:
                self._df = pd.concat([self._df, new_rows], ignore_index=True)

        return ids

    def delete_by_ids(self, data_ids: Union[str, List[str]]) -> int:
        """
        根据 data_id 精确删除数据 (支持单个 ID 或 ID 列表)。
        """
        if isinstance(data_ids, str):
            data_ids = [data_ids]

        with self._lock:
            if self._df.empty or self.ID_COL not in self._df.columns:
                return 0

            # 记录删除前的行数
            original_count = len(self._df)

            # 使用 ~ (取反) 和 isin 进行向量化过滤，性能极高
            self._df = self._df[~self._df[self.ID_COL].isin(data_ids)].reset_index(drop=True)

            # 返回实际删除的行数
            return original_count - len(self._df)

    def delete_data(self, filters: Dict[str, Any]) -> int:
        """
        根据条件字典删除数据。
        """
        with self._lock:
            if self._df.empty:
                return 0

            mask = pd.Series([True] * len(self._df))
            for col, val in filters.items():
                if col not in self._df.columns:
                    # 如果过滤条件里的列不存在，说明没有匹配数据
                    return 0
                mask = mask & (self._df[col] == val)

            rows_to_delete = mask.sum()
            if rows_to_delete > 0:
                self._df = self._df[~mask].reset_index(drop=True)

            return int(rows_to_delete)

    def query_data(self, filters: Dict[str, Any] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        查询数据。
        """
        with self._lock:
            if self._df.empty:
                return []

            # 如果没有 filters，默认查询所有
            if not filters:
                result_df = self._df
            else:
                mask = pd.Series([True] * len(self._df))
                for col, val in filters.items():
                    if col not in self._df.columns:
                        return []
                    mask = mask & (self._df[col] == val)
                result_df = self._df[mask]

            if limit is not None:
                result_df = result_df.head(limit)

            return result_df.to_dict(orient='records')

    def update_by_id(self, data_id: str, updates: Dict[str, Any]) -> bool:
        """
        根据 data_id 更新单条数据。

        Args:
            data_id: 目标数据的 ID
            updates: 需要更新的字段和值，例如 {'status': 'processed', 'score': 0.99}

        Returns:
            bool: 是否更新成功 (如果 ID 不存在则返回 False)
        """
        with self._lock:
            if self._df.empty:
                return False

            # 查找匹配的索引
            # 使用 .index 获取索引对象，比直接布尔索引取值更快
            mask = self._df[self.ID_COL] == data_id

            # 如果没有匹配的数据
            if not mask.any():
                return False

            # 执行更新
            # 注意：使用 .loc 避免 SettingWithCopyWarning
            for col, val in updates.items():
                self._df.loc[mask, col] = val

            return True

    def update_data(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        根据筛选条件批量更新数据。

        Args:
            filters: 筛选条件，例如 {'status': 'pending'}
            updates: 需要更新的内容

        Returns:
            int: 实际更新的行数
        """
        with self._lock:
            if self._df.empty:
                return 0

            # 构建布尔掩码
            mask = pd.Series([True] * len(self._df))
            for col, val in filters.items():
                if col not in self._df.columns:
                    # 如果过滤条件的列不存在，说明无法匹配任何数据
                    return 0
                mask = mask & (self._df[col] == val)

            rows_to_update = mask.sum()
            if rows_to_update == 0:
                return 0

            # 批量赋值
            for col, val in updates.items():
                self._df.loc[mask, col] = val

            return int(rows_to_update)
