"""配置中心：全项目唯一的配置入口。

工程化要点:
1. 所有可变参数(模型/路径/检索参数)集中在此, 通过 .env 覆盖, 改配置不改代码
2. pydantic-settings 启动时自动校验类型与合法性, 配置错误立刻暴露而不是运行中崩溃
3. lru_cache 单例, 保证进程内只加载一次
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置项(字段名与 .env 中的环境变量名一一对应)"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "企业智能RAG问答系统"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------- LLM(生成) ----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ---------- Embedding(向量化) ----------
    embedding_provider: str = "local"  # local | openai_compatible
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_model: str = "BAAI/bge-m3"

    # ---------- 重排序 ----------
    enable_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"

    # ---------- 检索 ----------
    top_k_retrieve: int = Field(default=8, ge=1, le=50)
    top_k_final: int = Field(default=4, ge=1, le=20)
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # ---------- 分块 ----------
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)

    # ---------- 存储 ----------
    chroma_dir: str = "data/chroma"
    raw_docs_dir: str = "data/raw_docs"
    collection_name: str = "enterprise_kb"

    # ---------- 会话 ----------
    max_history_messages: int = 20


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例(首次调用时读取 .env 并校验)"""
    return Settings()
