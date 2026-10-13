"""依赖注入: 组件单例的获取入口(FastAPI Depends体系)。

组件在应用启动时于 main.py 的 lifespan 中初始化一次,
此处仅提供访问入口, 保证全进程共享同一实例(向量化模型/连接只加载一次)。
"""
from functools import lru_cache

from app.agent.rag_agent import RAGAgent
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import HybridRetriever
from app.vectordb.store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_retriever() -> HybridRetriever:
    from app.embeddings.provider import create_embedder
    return HybridRetriever(create_embedder(), get_vector_store())


@lru_cache
def get_rag_agent() -> RAGAgent:
    from app.retrieval.reranker import create_reranker
    return RAGAgent(get_retriever(), create_reranker())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    from app.embeddings.provider import create_embedder
    return IngestionPipeline(create_embedder(), get_vector_store())
