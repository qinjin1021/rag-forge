"""文本清洗: 在分块前去除噪声, 提升分块质量与检索精度。

真实企业文档常见噪声: 全角/半角混排、多余空行、重复的页眉页脚、
异常控制字符等, 不清洗会直接污染向量。
"""


def clean_text(text: str) -> str:
    """执行全部清洗规则(纯函数, 便于单元测试)"""
    rules = (
        _normalize_chars,      # 全角字母数字/标点归一化
        _strip_control_chars,  # 去除控制符(保留换行)
        _collapse_spaces,      # 行内连续空格压缩
        _collapse_blank_lines, # 连续空行压缩为一个
        _dedup_lines,          # 去除连续重复行(页眉页脚残留)
    )
    for rule in rules:
        text = rule(text)
    return text.strip()


def _normalize_chars(text: str) -> str:
    """全角转半角(数字/字母/常见标点), 统一检索与LLM阅读体验"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:               # 全角空格
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:   # 全角字符区
            ch = chr(code - 0xFEE0)
        out.append(ch)
    return "".join(out)


def _strip_control_chars(text: str) -> str:
    """去除控制字符, 保留\n与\t"""
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def _collapse_spaces(text: str) -> str:
    """行内连续空白压缩为单个空格"""
    return "\n".join(
        " ".join(line.split()) if line.strip() else ""
        for line in text.split("\n")
    )


def _collapse_blank_lines(text: str) -> str:
    """连续空行压缩为一个段落分隔"""
    lines, out, blank = text.split("\n"), [], False
    for line in lines:
        if line.strip():
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out)


def _dedup_lines(text: str) -> str:
    """去除紧邻的完全重复行(常见于表格断页/页眉残留)"""
    lines = text.split("\n")
    out = [lines[0]] if lines else []
    for line in lines[1:]:
        if line.strip() and line.strip() == out[-1].strip():
            continue
        out.append(line)
    return "\n".join(out)
