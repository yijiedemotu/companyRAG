"""评测数据集：JSONL 载入、用例规范化、幂等入库。

**一句话定位**：把"标注好的一份考卷"变成 `eval_datasets` / `eval_cases` 两行数据，
并且**重复导入同一份文件不会产生重复数据**。

**在链路中的位置**：
``scripts/*.jsonl`` / `POST /eval/datasets` -> **本模块** -> `eval_datasets` + `eval_cases`
-> ``evaluation.harness`` 逐条跑 -> `eval_runs` / `eval_case_results`。

**关键设计取舍**：

1. **幂等键是 `EvalDataset.name` + `EvalCase.question`**（不是自增 id）。
   标注集会被反复修订（补题、改标准答案），如果每次导入都 `INSERT`，
   库里会堆积同一道题的多个版本，`recall@5` 的分母被无声放大——**指标失真且看不出来**。
   用问题文本做幂等键的代价是：**改题干会被当成一道新题**（旧题保留）。
   这是刻意的取舍——旧题对应的历史 `eval_case_results` 不能被牵连删除。
2. **`load_jsonl` 忽略空行与 `#` 注释行**：标注文件是人工维护的，
   需要支持"注释掉一条待确认的题"而不必真的删掉（保留上下文给下次讨论）。
3. **`encoding="utf-8-sig"`**：Windows 上手工用记事本/Excel 导出的文件常带 BOM，
   用 `utf-8` 读会让第一行的第一个字段名变成 `\ufeffquestion`，
   表现为"必填字段 question 缺失"——这种报错看着莫名其妙，所以统一按兼容 BOM 的方式读。
4. **本模块不做 `bm25_only_ablation`**：消融实验的编排属于 `harness.py`，
   本模块只负责"数据进得来"。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowflow.core.exceptions import ValidationError
from knowflow.core.logging import get_logger
from knowflow.db.models.evaluation import EvalCase, EvalDataset

logger = get_logger(__name__)

#: `expected_doc` 列长度上限（String(255)）。超长会被 MySQL 严格模式拒绝，
#: 所以在这里就截断，避免整份数据集导入失败。
_MAX_DOC_CHARS = 255


@dataclass
class CaseSpec:
    """一道评测题的内存形态（构造它不碰数据库，校验必须前置）。

    字段与 `EvalCase` 一一对应，但**不绑定 ORM**：数据集可以在没有数据库的
    单元测试里构造、比较、去重。
    """

    question: str
    ground_truth: str | None = None
    expected_doc: str | None = None
    expected_sections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """给 API 响应/日志用的普通 dict（不含 ORM 对象）。"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_doc": self.expected_doc,
            "expected_sections": list(self.expected_sections),
            "tags": list(self.tags),
        }


def _normalize_list(value: Any) -> list[str]:
    """把 JSON 里的"列表"字段规范化。

    接受三种真实存在的写法：`["a","b"]`、`"a"`（单值写成字符串）、`None`。
    统一成**去重且保序的 list[str]**：JSON 列里存 `None` 还是 `[]`
    会让下游 `or []` 与 `if value:` 两种写法行为不一致，索性在入口归一。
    """
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def case_from_dict(raw: dict[str, Any]) -> CaseSpec:
    """把一行 JSON 变成校验过的 :class:`CaseSpec`。

    必填只有 `question`（例如 `知识库支持哪些文件格式`）；`ground_truth`
    允许为空——"检索是否命中"不依赖标准答案，缺答案只影响生成指标。

    校验失败抛 `core.exceptions.ValidationError`（400 语义），
    由 API 层统一映射成错误响应，而不是让 `KeyError` 变成 500。
    """
    if not isinstance(raw, dict):
        raise ValidationError(f"评测用例必须是 JSON 对象，实际是 {type(raw).__name__}")

    question = str(raw.get("question") or "").strip()
    if not question:
        raise ValidationError("评测用例缺少必填字段 question（或为空字符串）")

    ground_truth = raw.get("ground_truth")
    expected_doc = raw.get("expected_doc")
    doc_text = str(expected_doc).strip() if expected_doc else None
    if doc_text and len(doc_text) > _MAX_DOC_CHARS:
        # 截断而不是报错：一个超长 doc 名不该让整份数据集导入失败
        doc_text = doc_text[:_MAX_DOC_CHARS]

    return CaseSpec(
        question=question,
        ground_truth=str(ground_truth).strip() if ground_truth else None,
        expected_doc=doc_text,
        # 兼容两种字段名：契约里是 expected_sections，人工标注时常写成 sections
        expected_sections=_normalize_list(
            raw.get("expected_sections") if "expected_sections" in raw else raw.get("sections")
        ),
        tags=_normalize_list(raw.get("tags")),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL：忽略空行与 `#` 开头的注释行，返回原始 dict 列表。

    用 `utf-8-sig` 读（见模块 docstring 第 3 条：BOM 会让第一个字段名带上
    不可见字符，报错信息极具误导性）。

    解析失败时**带上行号**抛 `ValidationError`：一份 500 行的标注文件里，
    没有行号的 JSONDecodeError 等于让人逐行找。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"评测数据集文件不存在: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{file_path.name} 第 {lineno} 行不是合法 JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValidationError(
                    f"{file_path.name} 第 {lineno} 行必须是 JSON 对象，"
                    f"实际是 {type(payload).__name__}"
                )
            rows.append(payload)
    logger.debug("eval.dataset.loaded", file=file_path.name, cases=len(rows))
    return rows


def get_or_create_dataset(
    session: Session,
    *,
    name: str,
    description: str | None = None,
    cases: Sequence[CaseSpec],
) -> EvalDataset:
    """按名字取数据集，不存在则创建；已存在则**按 question upsert 用例**。

    幂等实现细节：

    - 数据集本身按 `name`（唯一索引）查；
    - 用例按 `question` 逐条比对：命中就更新字段，未命中就插入；
    - **不做删除**：库里多出来的旧题保留（见模块 docstring 第 1 条），
      删除会 CASCADE 掉历史 `eval_case_results`，让"上次评测结果"凭空消失。

    返回更新后的 `EvalDataset`（`case_count` 已同步为真实条数）。
    **不 commit**：事务边界由调用方（service 层）决定。
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValidationError("数据集 name 不能为空")

    dataset = session.execute(
        select(EvalDataset).where(EvalDataset.name == clean_name)
    ).scalar_one_or_none()
    if dataset is None:
        dataset = EvalDataset(name=clean_name, description=description, case_count=0)
        session.add(dataset)
        session.flush()  # 需要 id 才能写 eval_cases.dataset_id

    if description is not None:
        dataset.description = description

    existing = session.execute(select(EvalCase).where(EvalCase.dataset_id == dataset.id)).scalars()
    by_question: dict[str, EvalCase] = {row.question: row for row in existing}

    inserted = 0
    updated = 0
    for spec in cases:
        row = by_question.get(spec.question)
        if row is None:
            row = EvalCase(dataset_id=dataset.id, question=spec.question)
            session.add(row)
            by_question[spec.question] = row
            inserted += 1
        else:
            updated += 1
        row.ground_truth = spec.ground_truth
        row.expected_doc = spec.expected_doc
        # 空列表存 None 而不是 []：JSON 列里两者语义相同，但统一成 None
        # 可以让 `case.tag_list` 的 `or []` 兜底在各个版本上行为一致
        row.expected_sections = list(spec.expected_sections) or None
        row.tags = list(spec.tags) or None

    session.flush()
    # case_count 是冗余计数（列表页要显示"共 N 条"）：必须在用例写完之后再统计
    dataset.case_count = len(by_question)
    logger.info(
        "eval.dataset.upserted",
        dataset=clean_name,
        inserted=inserted,
        updated=updated,
        total=dataset.case_count,
    )
    return dataset


def seed_dataset(
    session: Session,
    *,
    name: str,
    description: str,
    cases: Sequence[CaseSpec],
) -> tuple[int, int]:
    """seed 专用入口：返回 `(dataset_id, case_count)`。幂等（同名不重复插用例）。"""
    dataset = get_or_create_dataset(session, name=name, description=description, cases=cases)
    return int(dataset.id), int(dataset.case_count)


__all__ = [
    "CaseSpec",
    "case_from_dict",
    "get_or_create_dataset",
    "load_jsonl",
    "seed_dataset",
]
