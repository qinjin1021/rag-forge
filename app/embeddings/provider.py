"""Embedding模型抽象层(策略模式)。

工程化要点:
- 上层管道只依赖 BaseEmbedding 抽象接口, 不关心具体模型来源
- 两种实现可配置切换:
    * local             本地sentence-transformers, 免费/离线/内网可用(企业数据不出域)
    * openai_compatible 兼容OpenAI协议的Embedding API(硅基流动/智谱/通义等)
- 模型懒加载: 进程启动不加载模型, 首次调用才初始化, 加快服务启动与测试速度
"""
import os
from abc import ABC, abstractmethod
from typing import Sequence

from app.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 每批向量化的文本条数(API模式防止超限, 本地模式控制显存/内存)
BATCH_SIZE = 32


class BaseEmbedding(ABC):
    """Embedding统一接口"""

    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """文档分块向量化(入库时用)"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """查询向量化(检索时用)"""


class LocalEmbedding(BaseEmbedding):
    """本地sentence-transformers实现(默认), 模型文件缓存在 data/models"""

    def __init__(self, settings: Settings):
        import os as _os
        # 国内网络环境下默认走HF镜像, 若已显式配置则不覆盖
        _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        _os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "data/models")

        self._model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None  # 懒加载

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("加载本地Embedding模型: %s (首次运行会自动下载)", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            # sentence-transformers 6.x改名, 兼容新旧版本
            get_dim = getattr(self._model, "get_embedding_dimension", None) \
                or self._model.get_sentence_embedding_dimension
            self.dimension = get_dim()
            logger.info("模型就绪, 向量维度: %d", self.dimension)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        vectors = self._model.encode(
            list(texts), batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        return self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


class OpenAICompatibleEmbedding(BaseEmbedding):
    """OpenAI协议Embedding API实现(硅基流动/智谱/通义等均可)"""

    def __init__(self, settings: Settings):
        from openai import OpenAI

        if not settings.embedding_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai_compatible 时必须配置 EMBEDDING_API_KEY")
        self._client = OpenAI(
            api_key=settings.embedding_api_key, base_url=settings.embedding_base_url
        )
        self._model = settings.embedding_api_model
        self.dimension = settings.embedding_dimension

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[i:i + BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def create_embedder(settings: Settings | None = None) -> BaseEmbedding:
    """工厂函数: 按配置实例化Embedding实现"""
    settings = settings or get_settings()
    provider = settings.embedding_provider
    logger.info("创建Embedding组件: provider=%s", provider)
    if provider == "local":
        return LocalEmbedding(settings)
    if provider == "openai_compatible":
        return OpenAICompatibleEmbedding(settings)
    raise ValueError(f"未知的EMBEDDING_PROVIDER: {provider}(可选: local | openai_compatible)")
