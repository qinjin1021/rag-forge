"""知识库文档管理接口: 上传入库 / 列表 / 删除 / 全量重建。"""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.deps import get_ingestion_pipeline, get_retriever, get_vector_store
from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import DocumentInfo, IngestResult
from app.ingestion.loaders import SUPPORTED_EXTENSIONS

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["知识库管理"])

# 文件名安全化: 防路径穿越, 只保留常见安全字符
_UNSAFE_CHARS = re.compile(r"[^\w.\-\u4e00-\u9fa5]")


@router.post("/upload", response_model=list[IngestResult])
async def upload_documents(files: list[UploadFile]) -> list[IngestResult]:
    """上传文档并立即入库解析(支持多文件)。重复上传同名文档为覆盖更新。"""
    pipeline = get_ingestion_pipeline()
    retriever = get_retriever()
    settings = get_settings()
    docs_dir = Path(settings.raw_docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    results: list[IngestResult] = []
    for file in files:
        name = _UNSAFE_CHARS.sub("_", Path(file.filename or "unnamed").name)
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            results.append(IngestResult(
                source=name, chunk_count=0, duration_ms=0, success=False,
                error=f"不支持的格式(支持: {sorted(SUPPORTED_EXTENSIONS)})",
            ))
            continue
        save_path = docs_dir / name
        save_path.write_bytes(await file.read())
        logger.info("文档已保存: %s", save_path)
        results.append(pipeline.ingest_file(save_path))

    if any(r.success for r in results):
        retriever.refresh()  # 重建BM25索引, 使新文档立即可检索
    return results


@router.get("", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """列出知识库全部文档及分块数"""
    return [
        DocumentInfo(source=source, chunk_count=count)
        for source, count in get_vector_store().list_sources().items()
    ]


@router.delete("/{source}")
async def delete_document(source: str) -> dict:
    """从知识库删除指定文档(向量库分块 + 本地文件)"""
    store = get_vector_store()
    if source not in store.list_sources():
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")
    store.delete_by_source(source)
    file_path = Path(get_settings().raw_docs_dir) / source
    if file_path.is_file():
        file_path.unlink()
    get_retriever().refresh()
    logger.info("文档已删除: %s", source)
    return {"ok": True, "deleted": source}


@router.post("/ingest", response_model=list[IngestResult])
async def ingest_all() -> list[IngestResult]:
    """全量(重新)入库 data/raw_docs 目录下的所有文档"""
    pipeline = get_ingestion_pipeline()
    results = pipeline.ingest_directory()
    if any(r.success for r in results):
        get_retriever().refresh()
    return results
