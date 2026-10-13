"""分块器单元测试"""
from app.ingestion.splitters import RecursiveSplitter


def make_text(n_sentences: int, sentence: str = "这是用于测试分块逻辑的一句话。") -> str:
    return sentence * n_sentences


def test_short_text_single_chunk():
    """短文本不切分"""
    splitter = RecursiveSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split("年假5天。", "测试.md")
    assert len(chunks) == 1
    assert chunks[0].text == "年假5天。"
    assert chunks[0].source == "测试.md"
    assert chunks[0].total_chunks == 1


def test_long_text_chunk_sizes():
    """长文本切分后每块不超过chunk_size(允许overlap带来的少量超出边界由合并逻辑控制)"""
    splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split(make_text(100), "长文.md")
    assert len(chunks) > 1
    # 每块长度受控(chunk_size + overlap上限内)
    assert all(len(c.text) <= 200 + 40 for c in chunks)


def test_chunk_metadata_sequence():
    """块序号连续, 元数据完整, chunk_id稳定"""
    splitter = RecursiveSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split(make_text(60), "元数据.md")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "元数据.md" for c in chunks)
    assert all(c.chunk_id.startswith("元数据.md::chunk_") for c in chunks)


def test_sentence_boundary_preferred():
    """优先在句子边界切分: 每块结尾尽量是句号结尾"""
    splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split(make_text(20), "句子.md")
    assert sum(c.text.endswith("。") for c in chunks) >= len(chunks) - 1


def test_invalid_config_rejected():
    """overlap >= chunk_size 时应拒绝初始化"""
    import pytest
    with pytest.raises(AssertionError):
        RecursiveSplitter(chunk_size=100, chunk_overlap=100)
