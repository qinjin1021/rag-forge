"""摄取流水线编排: 把 加载/清洗/分块/向量化/入库 串成一条可复用的管道。

工程化要点:
- 每篇文档独立处理, 单个失败不影响其它文档(失败隔离)
- 稳定chunk_id + 先删后插, 重复摄取同一文档即"覆盖更新"(幂等)
"""
import time
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import IngestResult
from app.embeddings.provider import BaseEmbedding
from app.ingestion.cleaners import clean_text
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document
from app.ingestion.splitters import RecursiveSplitter
from app.vectordb.store import VectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """文档入库流水线: load -> clean -> split -> embed -> upsert"""

    def __init__(self, embedder: BaseEmbedding, store: VectorStore):
        settings = get_settings()
        self.embedder = embedder
        self.store = store
        self.splitter = RecursiveSplitter(settings.chunk_size, settings.chunk_overlap)

    def ingest_file(self, path: Path) -> IngestResult:
        """入库单个文档(幂等: 同名文档重新入库为覆盖更新)"""
        source = path.name
        start = time.perf_counter()
        try:
            # 1. 加载 -> 2. 清洗 -> 3. 分块
            text = clean_text(load_document(path))
            chunks = self.splitter.split(text, source)
            if not chunks:
                raise DocumentLoadError("清洗分块后无有效内容")
            # 4. 批量向量化 -> 5. 覆盖式入库
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.delete_by_source(source)
            self.store.upsert_chunks(chunks, vectors)
            duration = int((time.perf_counter() - start) * 1000)
            logger.info("入库成功: %s, %d块, %dms", source, len(chunks), duration)
            return IngestResult(source=source, chunk_count=len(chunks), duration_ms=duration)
        except DocumentLoadError as exc:
            logger.error("入库失败: %s, 原因: %s", source, exc)
            return IngestResult(
                source=source, chunk_count=0,
                duration_ms=int((time.perf_counter() - start) * 1000),
                success=False, error=str(exc),
            )

    def ingest_directory(self, directory: Path | None = None) -> list[IngestResult]:
        """批量入库目录下所有支持的文档"""
        directory = directory or Path(get_settings().raw_docs_dir)
        if not directory.is_dir():
            logger.warning("目录不存在: %s", directory)
            return []
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("批量入库开始: %s, 共%d个文件", directory, len(files))
        return [self.ingest_file(f) for f in files]
