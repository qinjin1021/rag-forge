"""统一数据模型(Pydantic)。

工程化要点:
- 摄取、检索、生成、API各层之间流转的数据结构在此统一定义,
  类型错误在边界处即被发现, 避免dict满天飞。
"""
from time import time
from typing import Optional

from pydantic import BaseModel, Field


# ==================== 摄取阶段 ====================
class Chunk(BaseModel):
    """文档分块: 摄取管道的最小处理单元"""

    chunk_id: str                 # 稳定ID(来源+序号), 用于增量更新/去重
    text: str                     # 分块正文
    source: str                   # 来源文件名, 用于引用溯源与按文档删除
    chunk_index: int = 0          # 在原文档中的序号
    total_chunks: int = 0         # 原文档总块数


class IngestResult(BaseModel):
    """单篇文档入库结果"""

    source: str
    chunk_count: int
    duration_ms: int
    success: bool = True
    error: Optional[str] = None


# ==================== 检索阶段 ====================
class RetrievedChunk(BaseModel):
    """检索命中的片段(带相关性得分)"""

    chunk_id: str
    text: str
    source: str
    chunk_index: int = 0
    score: float = 0.0            # 融合后的相关性得分(0~1)
    vector_score: float = 0.0     # 向量相似度单独保留, 便于调试
    bm25_score: float = 0.0       # 关键词得分单独保留


class SourceRef(BaseModel):
    """返回给前端的引用来源"""

    index: int                    # 引用编号, 与回答中的[1][2]对应
    source: str                   # 文档名
    snippet: str                  # 命中片段摘要
    score: float


# ==================== 对话阶段 ====================
class ChatMessage(BaseModel):
    """单条对话消息"""

    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """问答接口入参"""

    session_id: str = Field(default="default", description="会话ID, 支持多轮对话")
    query: str = Field(min_length=1, max_length=2000, description="用户问题")


class ChatResponse(BaseModel):
    """问答接口出参(非流式)"""

    answer: str
    sources: list[SourceRef] = []
    rewritten_query: str = ""     # 展示给用户看Agent如何理解了问题
    used_retrieval: bool = True   # 是否走了知识库检索(闲聊类问题为False)
    latency_ms: int = 0
    trace_id: str = ""


class DocumentInfo(BaseModel):
    """知识库中的文档信息"""

    source: str
    chunk_count: int
