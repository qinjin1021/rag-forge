"""统一日志配置 + 请求链路追踪(trace_id)。

工程化要点:
1. 全项目统一日志格式, 便于排查问题与接入ELK等日志平台
2. 通过 contextvars 实现每个请求独立的 trace_id, 一次问答的全链路日志可串联
"""
import logging
import uuid
from contextvars import ContextVar

# 当前请求的追踪ID(协程安全)
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    """生成并记录新的trace_id, 中间件在每个请求入口调用"""
    tid = uuid.uuid4().hex[:12]
    trace_id_var.set(tid)
    return tid


class TraceIdFilter(logging.Filter):
    """把contextvars中的trace_id注入到每条日志记录中"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def setup_logging(debug: bool = False) -> None:
    """初始化root logger, 进程内只需调用一次"""
    root = logging.getLogger()
    if root.handlers:  # 防止重复初始化
        return
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(TraceIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | [%(trace_id)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    # 降低第三方库噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
