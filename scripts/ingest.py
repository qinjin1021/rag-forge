"""命令行入库工具。

用法:
    python scripts/ingest.py                 # 入库 data/raw_docs 下全部文档
    python scripts/ingest.py --rebuild       # 清空向量库后全量重建
"""
import argparse
import shutil
import sys
from pathlib import Path

# 让脚本可以直接以 python scripts/xxx.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings                       # noqa: E402
from app.core.logging import get_logger, setup_logging    # noqa: E402
from app.embeddings.provider import create_embedder       # noqa: E402
from app.ingestion.pipeline import IngestionPipeline      # noqa: E402
from app.vectordb.store import VectorStore                # noqa: E402

logger = get_logger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库文档入库工具")
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后全量重建")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    if args.rebuild:
        logger.info("清空向量库: %s", settings.chroma_dir)
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)

    store = VectorStore()
    pipeline = IngestionPipeline(create_embedder(), store)
    results = pipeline.ingest_directory(Path(settings.raw_docs_dir))

    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    total_chunks = sum(r.chunk_count for r in ok)
    print("\n========== 入库结果 ==========")
    for r in results:
        mark = "✔" if r.success else "✘"
        extra = f" | {r.error}" if r.error else ""
        print(f" {mark} {r.source}: {r.chunk_count}块, {r.duration_ms}ms{extra}")
    print(f"==============================")
    print(f" 成功 {len(ok)}/{len(results)} 篇, 共{total_chunks}块, 库内总计{store.count()}块")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
