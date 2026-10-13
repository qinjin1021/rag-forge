"""Chroma向量库封装(仓储模式)。

工程化要点:
- 上层只调用本类方法, 不直接触碰chromadb API; 未来换Milvus/Qdrant只需重写本类
- 持久化到本地磁盘(data/chroma), 重启不丢数据
- 所有写操作使用upsert语义 + 按source删除, 保证文档级增量更新
"""
from pathlib import Path

import chromadb

from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)


class VectorStore:
    """Chroma集合的仓储封装"""

    def __init__(self):
        settings = get_settings()
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        # 嵌入式模式: 无需独立部署数据库进程, PersistentClient落盘
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离, 语义相似度标准选择
        )
        logger.info(
            "向量库就绪: dir=%s collection=%s 现有%d块",
            settings.chroma_dir, settings.collection_name, self._collection.count(),
        )

    # ---------------- 写操作 ----------------
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """批量写入/更新分块向量"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks与vectors数量不一致: {len(chunks)} != {len(vectors)}")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index, "total_chunks": c.total_chunks}
                for c in chunks
            ],
        )

    def delete_by_source(self, source: str) -> None:
        """删除某文档的全部分块(用于文档更新/下架)"""
        deleted = self._collection.get(where={"source": source})
        if deleted["ids"]:
            self._collection.delete(ids=deleted["ids"])
            logger.info("删除旧分块: %s (%d块)", source, len(deleted["ids"]))

    # ---------------- 读操作 ----------------
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """向量相似度检索。Chroma返回cosine距离, 转换为相似度=1-距离"""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=result["documents"][0][i],
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    score=1.0 - result["distances"][0][i],
                    vector_score=1.0 - result["distances"][0][i],
                )
            )
        return hits

    def get_all_chunks(self) -> list[RetrievedChunk]:
        """全量读取(BM25建索引用; 演示规模适用, 海量场景换Elasticsearch)"""
        if self.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=result["documents"][i],
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                )
            )
        return chunks

    def list_sources(self) -> dict[str, int]:
        """按文档统计分块数(知识库管理页用)"""
        counts: dict[str, int] = {}
        if self.count() == 0:
            return counts
        result = self._collection.get(include=["metadatas"])
        for meta in result["metadatas"]:
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def count(self) -> int:
        return self._collection.count()
