"""
semantic_chunker.py —— 语义切分

职责：将清洗后的长文本切分为知识库 chunk。
策略：标题层级优先（按「第x章 / x.x / #」等标题边界切），固定窗口兜底
      （超长无标题文本按 max_chars + overlap 滑窗切）。
所属链路：离线 A7。
设计：纯逻辑，不依赖外部模型；可对接 embedding 做「语义边界」微调（可选）。

说明：
  1) 标题层级优先，保证「一个规程条款 / 一个参数限额」尽量落在同一 chunk，
     便于检索命中后整段准确溯源。
  2) chunk_id 必须全局唯一。多份规程的页码与段落序号天然会重复（都从第 12 页
     第 0 段开始），若不携带文档标识，多文档入库时向量库主键会互相覆盖，
     表现为「刚入库的规程答不出来」。故 chunk_id 统一带上 doc_key 前缀。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from pipeline_offline.layout_analyzer import LayoutBlock


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    section: str = ""
    meta: dict = field(default_factory=dict)


# 标题层级正则（火电规程常见：第x章 / x.x.x / 数字. 开头）
_HEADING_RE = re.compile(r"(^|\n)\s*(第[一二三四五六七八九十百]+章|[\d]+(\.[\d]+){0,2})\s*[\.、]?\s*[一-龥]")


def _split_by_heading(text: str) -> List[str]:
    """按标题边界切分；若无标题则整体返回。"""
    matches = list(_HEADING_RE.finditer(text))
    if len(matches) <= 1:
        return [text]
    segs, prev = [], 0
    for m in matches:
        start = m.start()
        if prev:
            segs.append(text[prev:start].strip())
        prev = start
    segs.append(text[prev:].strip())
    return [s for s in segs if s]


def _sliding_window(text: str, max_chars: int, overlap: int) -> List[str]:
    """固定窗口兜底切分。"""
    if len(text) <= max_chars:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + max_chars])
        i += max_chars - overlap
    return out


def _make_chunk(block: LayoutBlock, text: str, n: int, doc_key: str = "") -> Chunk:
    """构造 chunk，chunk_id = {doc_key}_{文档内序号}，保证跨文档、跨版面块唯一。"""
    prefix = f"{doc_key}_" if doc_key else ""
    return Chunk(
        chunk_id=f"{prefix}{n}",
        text=text,
        page=block.page,
        section=block.meta.get("section", ""),
        meta=block.meta,
    )


def chunk_block(block: LayoutBlock, max_chars: int = 800, overlap: int = 120,
                doc_key: str = "", seq_offset: int = 0) -> List[Chunk]:
    """
    对单个 LayoutBlock 做语义切分：
      1) 标题层级优先；
      2) 切后超长或无法按标题切的，用固定窗口兜底。

    doc_key   ：文档标识（通常取文件名）。多份文档入库时必须传，
                否则页码+序号相同的 chunk 会在向量库里撞主键。
    seq_offset：文档级起始序号。一份文档含多个版面块（正文/表格/图片），
                每块各自从 0 编号必然撞车，故由调用方累加偏移。
    同一输入产出稳定 id，便于复现与排障。
    """
    parts = _split_by_heading(block.content)
    chunks: List[Chunk] = []
    n = seq_offset
    for part in parts:
        pieces = _sliding_window(part, max_chars, overlap) if len(part) > max_chars else [part]
        for piece in pieces:
            chunks.append(_make_chunk(block, piece, n, doc_key))
            n += 1
    return chunks
