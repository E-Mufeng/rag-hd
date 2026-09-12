"""
table_parser.py —— 表格解析（独立成块 + 元数据）

职责：把版面分析得到的表格区块解析为「独立 chunk」，并附加元数据
      （页码 page、表号 table_no、所属章节等），便于检索时精准溯源。
所属链路：离线 A3。
对接抽象接口：TableParser（业务上层只依赖该接口）。

【仅框架演示，真实部署替换】
    真实实现可对接内网表格识别服务 / 复杂表结构还原（合并单元格、跨页表）。
    本框架提供 MockTableParser，返回结构化表格文本 + 元数据。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from pipeline_offline.layout_analyzer import LayoutBlock


@dataclass
class TableChunk:
    chunk_id: str
    text: str             # 表格转为可检索文本（含表头与行列）
    page: int
    table_no: str
    meta: dict = field(default_factory=dict)


class TableParser(ABC):
    """表格解析抽象接口。"""

    @abstractmethod
    def parse(self, block: LayoutBlock) -> TableChunk:
        """将一个表格 LayoutBlock 解析为独立 TableChunk。"""
        ...


class MockTableParser(TableParser):
    """
    模拟表格解析（仅框架演示，真实部署替换）。

    将制表符/换行分隔的伪表格转为「表号 + 行列」文本，并保留 page/table_no 元数据。
    """

    def parse(self, block: LayoutBlock) -> TableChunk:
        rows = [r for r in block.content.split("\n") if r.strip()]
        # 将首行作为表头，构建可读文本
        readable = " | ".join(rows) if rows else block.content
        text = f"【{block.meta.get('table_no', '表')}】{readable}"
        return TableChunk(
            chunk_id=f"tbl_{block.page}_{block.meta.get('table_no', 'x')}",
            text=text,
            page=block.page,
            table_no=block.meta.get("table_no", "unknown"),
            meta=block.meta,
        )
