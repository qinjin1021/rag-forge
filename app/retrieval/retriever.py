"""混合检索器: 向量语义检索 + BM25关键词检索, 加权融合(RRF思想的简化实现)。

为什么要混合检索(企业落地的关键经验):
- 纯向量检索: 擅长"语义相近"(同义改写), 但对 专有名词/型号/人名 等精确词不敏感
- 纯BM25检索: 擅长精确关键词命中, 但不理解"报销标准"和"住宿限额"是一个意思
- 融合两者: 召回率和精度都显著优于单一方式

评分归一化: 两路得分除以各自最大值映射到[0,1], 再按权重加权求和。
"""
import threading

import jieba
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import RetrievedChunk
from app.embeddings.provider import BaseEmbedding
from app.vectordb.store import VectorStore

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """中文分词(BM25检索的基本单元), 去除空白token"""
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


def weighted_fuse(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    vector_weight: float,
    bm25_weight: float,
) -> list[RetrievedChunk]:
    """两路检索结果加权融合(纯函数, 便于单元测试)。

    规则: 每路得分按该路最高分归一化到[0,1], 未命中该路的按0分参与,
          融合分 = w_v*vec_norm + w_b*bm_norm, 按融合分降序。
    """
    max_vec = max((h.vector_score for h in vector_hits), default=0.0) or 1.0
    max_bm = max((h.bm25_score for h in bm25_hits), default=0.0) or 1.0

    merged: dict[str, RetrievedChunk] = {}
    for hit in vector_hits:
        merged[hit.chunk_id] = hit.model_copy(update={
            "vector_score": hit.vector_score / max_vec,
            "bm25_score": 0.0,
        })
    for hit in bm25_hits:
        if hit.chunk_id in merged:
            merged[hit.chunk_id].bm25_score = hit.bm25_score / max_bm
        else:
            merged[hit.chunk_id] = hit.model_copy(update={
                "bm25_score": hit.bm25_score / max_bm,
                "vector_score": 0.0,
            })
    for hit in merged.values():
        hit.score = vector_weight * hit.vector_score + bm25_weight * hit.bm25_score
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)


class HybridRetriever:
    """向量 + BM25 混合检索器"""

    def __init__(self, embedder: BaseEmbedding, store: VectorStore):
        self.settings = get_settings()
        self.embedder = embedder
        self.store = store
        # BM25内存索引(线程锁保护, 支持多worker并发检索)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[RetrievedChunk] = []
        self._lock = threading.Lock()

    # ---------------- 索引管理 ----------------
    def refresh(self) -> int:
        """从向量库全量重建BM25索引。入库/删除文档后调用。"""
        with self._lock:
            chunks = self.store.get_all_chunks()
            if chunks:
                corpus = [tokenize(c.text) for c in chunks]
                self._bm25 = BM25Okapi(corpus)
                self._bm25_chunks = chunks
            else:
                self._bm25, self._bm25_chunks = None, []
            logger.info("BM25索引已重建: %d块", len(chunks))
            return len(chunks)

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """混合检索主入口, 返回融合排序后的top_k片段"""
        top_k = top_k or self.settings.top_k_retrieve
        if self.store.count() == 0:
            return []

        recall_k = max(top_k * 2, 10)  # 每路多召回一些, 给融合留空间
        vector_hits = self.store.search(self.embedder.embed_query(query), recall_k)
        bm25_hits = self._bm25_search(query, recall_k)

        fused = weighted_fuse(
            vector_hits, bm25_hits,
            self.settings.vector_weight, self.settings.bm25_weight,
        )
        results = fused[:top_k]
        logger.info(
            "混合检索: query=%r 向量%d条 BM25%d条 融合后%d条 最高分%.3f",
            query, len(vector_hits), len(bm25_hits), len(results),
            results[0].score if results else 0.0,
        )
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25关键词检索(索引为空时返回空)"""
        with self._lock:
            bm25, chunks = self._bm25, self._bm25_chunks
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits: list[RetrievedChunk] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break  # 后续得分更低, 直接截断
            hits.append(chunks[idx].model_copy(update={"bm25_score": float(scores[idx])}))
        return hits
