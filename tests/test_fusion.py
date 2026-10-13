"""混合检索融合算法与分词测试"""
from app.core.schemas import RetrievedChunk
from app.retrieval.retriever import tokenize, weighted_fuse


def _hit(cid: str, vec: float = 0.0, bm: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=f"text-{cid}", source="doc.md",
        score=0.0, vector_score=vec, bm25_score=bm,
    )


def test_fuse_merges_union():
    """两路结果的并集都应保留"""
    vec_hits = [_hit("a", vec=0.9), _hit("b", vec=0.8)]
    bm_hits = [_hit("b", bm=5.0), _hit("c", bm=3.0)]
    fused = weighted_fuse(vec_hits, bm_hits, vector_weight=0.7, bm25_weight=0.3)
    assert {f.chunk_id for f in fused} == {"a", "b", "c"}


def test_fuse_ranking():
    """双路都命中的块应排在只命中一路的前面"""
    vec_hits = [_hit("both", vec=0.9), _hit("only_vec", vec=0.85)]
    bm_hits = [_hit("both", bm=8.0), _hit("only_bm", bm=6.0)]
    fused = weighted_fuse(vec_hits, bm_hits, vector_weight=0.5, bm25_weight=0.5)
    assert fused[0].chunk_id == "both"
    # 归一化后: both = 0.5*1.0 + 0.5*1.0 = 1.0
    assert abs(fused[0].score - 1.0) < 1e-6


def test_fuse_empty_inputs():
    """任一路为空时不崩溃, 返回另一路"""
    vec_hits = [_hit("a", vec=0.7)]
    fused = weighted_fuse(vec_hits, [], vector_weight=0.7, bm25_weight=0.3)
    assert len(fused) == 1
    assert fused[0].chunk_id == "a"


def test_fuse_normalization():
    """得分按各路最大值归一化到[0,1]"""
    vec_hits = [_hit("a", vec=0.6), _hit("b", vec=0.3)]
    fused = weighted_fuse(vec_hits, [], vector_weight=1.0, bm25_weight=0.0)
    assert abs(fused[0].vector_score - 1.0) < 1e-6  # 最高分归一化为1
    assert abs(fused[1].vector_score - 0.5) < 1e-6


def test_tokenize_chinese():
    """中文分词: 小写化 + 去空白token + 产出多个词"""
    tokens = tokenize("年假 有几天")
    assert len(tokens) >= 2
    assert all(t.strip() for t in tokens)
    assert "".join(tokens) == "年假有几天"
