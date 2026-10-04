"""把源码 docstring 里的 `>>>` 示例当成测试跑。

**为什么值得单独一个测试文件**：docstring 里的示例是**文档里最容易被信任、
也最容易过期**的东西 —— 读者会把它当"作者验过的例子"。

本项目真实翻过车：`retrieval/text.py` 的 `tokenize_zh` docstring 里写着

    >>> tokenize_zh("一线城市住宿标准")
    ['一', '线', '城', '市', ...]            ← 实际 `一` 会被 STOPWORDS 过滤掉
    >>> tokenize_zh("错误码 PAYLOAD_TOO_LARGE")
    ['错', '误', '码', 'payload_too_large']   ← 实际 `_` 是分隔符，会切成 3 个词

两处**都与实跑不一致**（是照"理想输出"手写的）。
它是被写教程的 agent 用实跑对照时才发现的 —— **靠人眼 review 是发现不了的**。

现在这两个模块的 docstring 示例由本测试**真的执行一遍**：
写错就红，改代码忘了改文档也会红。

⚠ 覆盖范围**刻意只限于"纯函数、无 IO、无外部依赖"的模块**（当前是
`llm/tokenizer.py` 与 `retrieval/text.py`）。docstring 里出现数据库、模型、
网络调用的模块**不适合**塞进 doctest —— 那会把"文档校验"变成"集成测试"，
既慢又脆。这类模块的 docstring 示例应当写成"带 `# 预期：` 注释的示意"，
并由对应的集成测试来保证正确性。
"""

from __future__ import annotations

import doctest
import importlib

import pytest

#: 需要跑 doctest 的模块（都满足：纯函数 / 无 IO / 无外部依赖）
DOCTEST_MODULES = [
    "knowflow.llm.tokenizer",
    "knowflow.retrieval.text",
]


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_docstring示例与实跑一致(module_name: str) -> None:
    """模块 docstring 里的每个 `>>>` 都必须真的跑出它写的结果。"""
    module = importlib.import_module(module_name)
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    # `failed == 0` 才算过；同时要求**确实找到了示例**，
    # 否则"删掉所有 `>>>` 让测试变绿"就成了绕过方式。
    assert result.attempted > 0, (
        f"{module_name} 里没找到任何 doctest 示例。"
        f"如果是有意删除的，请同时把它从 DOCTEST_MODULES 里移除 —— "
        f"否则这个参数化用例会变成一条「永远通过」的假保护。"
    )
    assert result.failed == 0, f"{module_name} 有 {result.failed} 个 docstring 示例与实跑不符"
