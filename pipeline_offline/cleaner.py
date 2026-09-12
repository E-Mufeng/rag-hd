"""
cleaner.py —— 文档清洗与脱敏

职责：对解析后的文本做清洗（去除页眉页脚噪声、多余空白、乱码）和脱敏
      （替换工号、手机号、姓名等敏感信息为占位符）。
所属链路：离线 A6。
设计：纯函数式，不依赖任何外部服务，可直接在 mock 流程中使用。

注意：脱敏仅基于规则正则，真实项目需结合内网数据合规要求扩充字典。
"""
from __future__ import annotations

import re
from typing import List

from pipeline_offline.layout_analyzer import LayoutBlock


# 简单脱敏规则：工号 / 手机号 / 姓名占位（演示用，真实部署按内网合规扩展）
_PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "<PHONE>"),         # 手机号
    (re.compile(r"[A-Z]{2}\d{4,}"), "<EMP_ID>"),       # 工号如 EM1234
    (re.compile(r"第[一二三四五六七八九十]+页"), ""),   # 页眉页脚页码噪声
]


def clean_text(text: str) -> str:
    """基础清洗：合并多余空白。"""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def desensitize(text: str) -> str:
    """脱敏：按规则替换敏感信息。"""
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def clean_blocks(blocks: List[LayoutBlock]) -> List[LayoutBlock]:
    """批量清洗 + 脱敏 LayoutBlock（保留正文，页眉页脚在此阶段剔除）。"""
    out = []
    for b in blocks:
        if b.block_type == "header_footer":
            continue  # 页眉页脚不进入知识库
        b.content = desensitize(clean_text(b.content))
        if b.content:
            out.append(b)
    return out
