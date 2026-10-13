"""会话记忆存储(多轮对话支持)。

演示版用进程内存实现; 接口与存储分离, 生产环境替换为Redis实现即可
(企业落地时多实例部署, 内存态无法跨节点共享, 这是标准演进路径)。
"""
from app.config import get_settings
from app.core.schemas import ChatMessage


class SessionStore:
    """按session_id维护对话历史, 自动截断防止无限增长"""

    def __init__(self, max_messages: int | None = None):
        self.max_messages = max_messages or get_settings().max_history_messages
        self._sessions: dict[str, list[ChatMessage]] = {}

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        messages = self._sessions.setdefault(session_id, [])
        messages.append(ChatMessage(role=role, content=content))
        # 只保留最近N条(0截断, 保证历史条数为偶数, 不破坏问答配对)
        if len(messages) > self.max_messages:
            self._sessions[session_id] = messages[-self.max_messages:]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
