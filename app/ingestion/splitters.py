"""递归字符分块器(自研实现, 不依赖langchain, 逻辑可控可测试)。

为什么需要分块:
- LLM上下文有限, 且长文档整体向量化会稀释语义, 检索不准
- 目标: 块内语义完整(尽量在段落/句子边界切), 块间保留overlap防止边界信息丢失

递归策略: 优先按 段落 -> 句子 -> 字符 逐级尝试切分,
        优先级高的分隔符能保持语义边界, 都不行时才硬切。
"""
from app.core.logging import get_logger
from app.core.schemas import Chunk

logger = get_logger(__name__)

# 分隔符优先级: 从"语义边界最强"到"最弱"
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "!", "?", ";", " ", ""]


class RecursiveSplitter:
    """递归字符分块器"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        assert 0 <= chunk_overlap < chunk_size, "overlap必须小于chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, source: str) -> list[Chunk]:
        """把清洗后的长文本切分为带元数据的分块列表"""
        pieces = self._split_recursive(text, self.chunk_size, _SEPARATORS)
        # 合并过碎的块并携带overlap
        merged = self._merge_with_overlap(pieces)
        chunks = [
            Chunk(
                chunk_id=f"{source}::chunk_{i}",
                text=t.strip(),
                source=source,
                chunk_index=i,
                total_chunks=len(merged),
            )
            for i, t in enumerate(merged)
            if t.strip()
        ]
        logger.info("分块完成: %s -> %d块", source, len(chunks))
        return chunks

    # ---------------- 内部实现 ----------------
    def _split_recursive(self, text: str, size: int, separators: list[str]) -> list[str]:
        """按分隔符优先级递归切分, 返回不超过size的文本片段"""
        if len(text) <= size:
            return [text]

        # 选出文本中实际存在的第一个分隔符
        sep = separators[-1]
        rest = separators
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep, rest = s, separators[i + 1:]
                break

        if sep == "":  # 无任何分隔符可用 -> 按长度硬切
            return [text[i:i + size] for i in range(0, len(text), size)]

        parts = text.split(sep)
        # 分隔符保留在片段末尾(如句号/换行), 保持句子完整与标点信息
        pieces: list[str] = []
        for i, part in enumerate(parts):
            fragment = part + (sep if i < len(parts) - 1 else "")
            if not fragment:
                continue
            if len(fragment) <= size:
                pieces.append(fragment)
            else:
                pieces.extend(self._split_recursive(fragment, size, rest))
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """把碎片段落合并到接近chunk_size, 并为相邻块添加overlap"""
        chunks: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = f"{buf}{piece}"
            if len(candidate) > self.chunk_size and buf:
                chunks.append(buf)
                # overlap: 保留上一块尾部, 让下一块携带上文
                buf = buf[-self.chunk_overlap:] + piece if self.chunk_overlap else piece
            else:
                buf = candidate
        if buf.strip():
            chunks.append(buf)
        return chunks
