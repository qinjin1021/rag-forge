"""重排序器(Cross-Encoder, 可选组件)。

为什么检索后还要重排:
- 初检(向量/BM25)是"双塔"式粗筛: 查询和文档独立编码, 速度快但精度有限
- Cross-Encoder把(查询, 文档)拼在一起深度交互打分, 精度高但慢
- 经典漏斗架构: 粗筛召回 top 8~20 -> 精排取 top 3~5 送入LLM

懒加载: 只有配置开启且真正调用时才下载/加载模型。
"""
import math
from abc import ABC, abstractmethod

from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import RetrievedChunk

logger = get_logger(__name__)


class BaseReranker(ABC):
    """重排序统一接口(便于未来替换为API版rerank, 如硅基流动bge-reranker)"""

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """按查询与片段的相关性重新排序, 返回前top_k"""


class CrossEncoderReranker(BaseReranker):
    """本地cross-encoder重排(bge-reranker系列, 中文效果好)"""

    def __init__(self, model_name: str):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "data/models")
        self._model_name = model_name
        self._model = None  # 懒加载

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("加载重排序模型: %s (首次运行会自动下载)", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        self._ensure_model()
        pairs = [(query, c.text[:1000]) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk, score in ranked:
            # sigmoid归一化到0~1, 与融合分数量纲对齐
            results.append(chunk.model_copy(update={"score": float(1 / (1 + math.exp(-score)))}))
        logger.info("重排完成: %d -> %d 条", len(chunks), len(results))
        return results


class NoopReranker(BaseReranker):
    """空实现: 未开启重排时直接截断, 保持调用方代码不变(空对象模式)"""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


def create_reranker():
    """工厂函数: 按配置返回重排器"""
    settings = get_settings()
    if settings.enable_rerank:
        return CrossEncoderReranker(settings.rerank_model)
    return NoopReranker()
