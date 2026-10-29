import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 将 early_stop 包所在目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# mock openai 依赖（CI 环境可能未安装 openai）
_mock_openai = MagicMock()
_mock_openai.types.responses.EasyInputMessage = MagicMock(
    side_effect=lambda content, role, type: MagicMock(content=content, role=role, type=type)
)
sys.modules.setdefault("openai", _mock_openai)
sys.modules.setdefault("openai.types", _mock_openai.types)
sys.modules.setdefault("openai.types.responses", _mock_openai.types.responses)