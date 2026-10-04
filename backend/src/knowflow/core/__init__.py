"""基础设施层：配置、日志、异常、安全。

`core` **不允许** import 任何其他业务包（只允许被 import）。
因为它是所有人的依赖，一旦反向依赖就会形成循环。
"""

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import KnowFlowError

__all__ = ["KnowFlowError", "Settings", "get_settings"]
