"""清洗器单元测试(纯离线, 无外部依赖)"""
from app.ingestion.cleaners import clean_text


def test_fullwidth_to_halfwidth():
    """全角字母数字与标点应被归一化为半角"""
    assert clean_text("２０２５年Ｑ１") == "2025年Q1"
    assert clean_text("（１）（２）") == "(1)(2)"


def test_collapse_spaces():
    """行内连续空格压缩为单个"""
    assert clean_text("报销    标准    400元") == "报销 标准 400元"


def test_collapse_blank_lines():
    """连续空行压缩为一个段落分隔, 首尾空白去除"""
    assert clean_text("第一段\n\n\n\n第二段") == "第一段\n\n第二段"
    assert clean_text("\n\n  开头结尾  \n\n") == "开头结尾"


def test_dedup_consecutive_lines():
    """紧邻重复行(页眉残留)被去除, 非紧邻重复保留"""
    assert clean_text("第1页\n第1页\n正文内容") == "第1页\n正文内容"
    assert clean_text("重要\n其他内容\n重要") == "重要\n其他内容\n重要"


def test_control_chars_removed():
    """控制字符被清除, 换行保留"""
    assert clean_text("带\x00\x01控制符\n的文本") == "带控制符\n的文本"
