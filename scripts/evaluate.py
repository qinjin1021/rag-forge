"""RAG效果评估脚本: 用标注好的评测集量化系统质量。

为什么需要评估(企业落地核心环节):
- 改了分块参数/换模型/调权重后, 系统是变好还是变坏? 不能靠感觉, 必须有可复现的量化指标
- 指标设计:
    1. 检索命中率(hit@k): 期望命中的文档是否出现在检索结果的top-k来源中 -> 衡量检索层
    2. 回答关键词覆盖率: 标注答案的关键信息点是否出现在生成回答中 -> 衡量端到端
    3. 平均延迟 -> 衡量体验

评测集格式(data/eval.jsonl, 每行一条):
    {"question": "...", "expected_source": "员工手册.md", "expected_keywords": ["5天", "10天"]}

用法:
    python scripts/evaluate.py              # 全量评估(检索+生成)
    python scripts/evaluate.py --retrieval  # 只评检索层(不调LLM, 免费快速)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings                      # noqa: E402
from app.core.logging import get_logger, setup_logging   # noqa: E402
from app.embeddings.provider import create_embedder      # noqa: E402
from app.retrieval.reranker import create_reranker       # noqa: E402
from app.retrieval.retriever import HybridRetriever      # noqa: E402
from app.vectordb.store import VectorStore               # noqa: E402

logger = get_logger("evaluate")


def load_eval_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"评测集不存在: {path}")
        sys.exit(1)
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"加载评测集: {len(samples)} 条\n")
    return samples


def evaluate_retrieval(retriever: HybridRetriever, reranker, samples: list[dict], settings) -> None:
    """检索层评估: 命中率 hit@k"""
    hits, rows = 0, []
    for s in samples:
        chunks = reranker.rerank(
            s["question"], retriever.retrieve(s["question"]), settings.top_k_final)
        sources = {c.source for c in chunks}
        hit = s.get("expected_source") in sources
        hits += hit
        rows.append((s["question"], hit, ", ".join(sorted(sources))))

    print("========== 检索层评估 (hit@top{}) ==========".format(settings.top_k_final))
    for q, hit, sources in rows:
        mark = "✔" if hit else "✘"
        print(f" {mark} {q}")
        print(f"    命中来源: {sources or '无'}")
    rate = hits / len(samples) * 100
    print(f"-------------------------------------------")
    print(f" 命中率: {hits}/{len(samples)} = {rate:.1f}%\n")


async def evaluate_end_to_end(samples: list[dict], settings) -> None:
    """端到端评估: 关键词覆盖率 + 延迟(需要LLM API)"""
    from app.agent.rag_agent import RAGAgent

    retriever = HybridRetriever(create_embedder(), VectorStore())
    retriever.refresh()
    agent = RAGAgent(retriever, create_reranker())

    covered_total, keyword_total, latencies, rows = 0, 0, [], []
    for s in samples:
        t0 = time.perf_counter()
        try:
            resp = await agent.chat(f"eval-{id(s)}", s["question"])
        except Exception as exc:
            print(f" ✘ {s['question']} -> 调用失败: {exc}")
            continue
        latencies.append(time.perf_counter() - t0)
        keywords = s.get("expected_keywords", [])
        covered = [k for k in keywords if k in resp.answer]
        covered_total += len(covered)
        keyword_total += len(keywords)
        rows.append((s["question"], resp.answer, len(covered), len(keywords), resp.latency_ms))

    print("========== 端到端评估 (关键词覆盖率) ==========")
    for q, answer, c, t, ms in rows:
        mark = "✔" if c == t else "△"
        print(f" {mark} {q} [{c}/{t}关键点, {ms}ms]")
        print(f"    回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
    if keyword_total:
        print(f"----------------------------------------------")
        print(f" 关键词覆盖率: {covered_total}/{keyword_total} = "
              f"{covered_total / keyword_total * 100:.1f}%")
    if latencies:
        print(f" 平均端到端延迟: {sum(latencies) / len(latencies):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG效果评估")
    parser.add_argument("--retrieval", action="store_true", help="只评估检索层(不调用LLM)")
    parser.add_argument("--dataset", default="data/eval.jsonl", help="评测集路径")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    samples = load_eval_dataset(Path(args.dataset))

    store = VectorStore()
    if store.count() == 0:
        print("知识库为空, 请先执行: python scripts/ingest.py")
        sys.exit(1)
    retriever = HybridRetriever(create_embedder(), store)
    retriever.refresh()
    reranker = create_reranker()

    evaluate_retrieval(retriever, reranker, samples, settings)
    if not args.retrieval:
        asyncio.run(evaluate_end_to_end(samples, settings))


if __name__ == "__main__":
    main()
