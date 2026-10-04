"""会话记忆：双重截断 + 数据库级并发安全。

**为什么必须截断**（两个都要，取更严者）：

| 只按轮数截断 | 只按字符截断 |
| --- | --- |
| 用户粘贴一篇 3000 字的需求后，10 轮就是 3 万字，照样爆上下文 | 如果每轮都很短（"嗯"、"继续"），会保留上百轮，语义上早已跑题 |

所以实现是「**从最新往回装，装不下就停**」：
既限制轮数（`MEMORY_MAX_TURNS`），也限制总字符（`MEMORY_MAX_CHARS`），
任何一个到顶就停止回溯。这样成本上限是可算的：
`prompt_token ≈ 记忆字符 + 上下文预算`，不会随对话轮数线性增长。

**为什么用 `with_for_update()` 而不是 Python 的 `threading.Lock`**：
Python 锁只在单进程内有效。生产上多 worker / 多副本时，
两个请求同时写同一会话会丢掉一次 `message_count` 自增，
表现为"会话列表显示的消息数比实际少" —— 难查且不影响功能，属于最烦的一类 bug。
`SELECT ... FOR UPDATE` 把并发控制交给数据库，跨进程也正确。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.db.models.chat import MODE_AGENT, Conversation, Message
from knowflow.llm.base import ChatMessage

logger = get_logger(__name__)

# 哪些角色参与"记忆"。tool/system 消息不进记忆：
# tool 消息是中间产物（已经体现在最终答案里），system 是每轮都会重新拼的。
MEMORY_ROLES = ("user", "assistant")


def truncate_history(
    messages: Sequence[ChatMessage], *, max_turns: int, max_chars: int
) -> list[ChatMessage]:
    """从最新往回装，同时受轮数与字符两个上限约束。返回**按时间正序**的列表。"""
    if not messages:
        return []

    kept: list[ChatMessage] = []
    used_chars = 0
    turns = 0

    for message in reversed(messages):
        if message.role not in MEMORY_ROLES:
            continue
        cost = len(message.content)
        # 至少保留最后一轮：否则会出现"有记忆配置但完全没记忆"的情况，
        # 多轮问题（"那二线城市呢？"）直接失效。
        if kept and (turns >= max_turns or used_chars + cost > max_chars):
            break
        if message.role == "user":
            turns += 1
        kept.append(message)
        used_chars += cost

    kept.reverse()
    return kept


def summarize_history(messages: Sequence[ChatMessage], *, limit: int = 400) -> str:
    """把最近几轮压成一行，用于改写查询时补全指代词（"它"、"那个标准"）。"""
    recent = [m for m in messages if m.role in MEMORY_ROLES][-4:]
    if not recent:
        return ""
    text = " / ".join(
        f"{'用户' if m.role == 'user' else '助手'}：{m.content.strip()[:120]}" for m in recent
    )
    return text[:limit]


def load_history(
    session: Session,
    conversation_id: int,
    *,
    settings: Settings | None = None,
) -> list[ChatMessage]:
    """从 MySQL 读取并截断会话历史。

    **为什么要多取一些再截断**（`limit = max_turns * 3`）：
    截断是按"字符预算从后往前装"，如果 SQL 只取最后 `max_turns` 条，
    当最后几条恰好都很长时，能装下的轮数会比配置少很多。
    多取一点（有上限，不会失控）让截断逻辑自己决定实际保留多少。
    """
    cfg = settings or get_settings()
    fetch_limit = max(cfg.memory_max_turns * 3, 10)

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role.in_(MEMORY_ROLES))
        .order_by(Message.id.desc())
        .limit(fetch_limit)
    )
    rows = list(session.execute(stmt).scalars())
    rows.reverse()

    raw = [ChatMessage(role=_role(m.role), content=m.content) for m in rows]
    return truncate_history(raw, max_turns=cfg.memory_max_turns, max_chars=cfg.memory_max_chars)


def _role(value: str) -> Any:
    return value if value in MEMORY_ROLES else "user"


def append_message(
    session: Session,
    *,
    conversation_id: int,
    role: str,
    content: str,
    **fields: Any,
) -> Message:
    """写入一条消息并原子地自增会话计数。

    `with_for_update()` 锁住会话行 —— 同一会话的并发写入会串行化，
    保证 `message_count` 与实际条数一致。
    """
    conversation = session.get(Conversation, conversation_id, with_for_update=True)
    if conversation is None:
        raise ValueError(f"会话 {conversation_id} 不存在")

    message = Message(conversation_id=conversation_id, role=role, content=content, **fields)
    session.add(message)
    conversation.message_count = (conversation.message_count or 0) + 1
    session.flush()
    return message


def touch_conversation_title(
    conversation: Conversation, *, question: str, max_len: int = 30
) -> None:
    """首轮提问时用问题当标题（比"新会话"有用得多）。已有自定义标题则不动。"""
    if conversation.title and conversation.title != "新会话":
        return
    title = " ".join(question.split())
    conversation.title = title[:max_len] + ("…" if len(title) > max_len else "")


def create_conversation(
    session: Session,
    *,
    user_id: int,
    kb_id: int | None,
    mode: str = MODE_AGENT,
    title: str = "新会话",
) -> Conversation:
    conversation = Conversation(user_id=user_id, kb_id=kb_id, mode=mode, title=title)
    session.add(conversation)
    session.flush()
    return conversation


__all__ = [
    "MEMORY_ROLES",
    "append_message",
    "create_conversation",
    "load_history",
    "summarize_history",
    "touch_conversation_title",
    "truncate_history",
]
