"""
layout_analyzer.py —— 版面分析抽象接口

职责：定义统一的版面分析抽象基类，区分【正文 / 表格 / 图片 / 页眉页脚】，
      并提供 OCR 适配器抽象（扫描件前置）。
所属链路：离线 A2（版面分析）、A3（表格）、A4（图片保留）。
对接实现：
    - 真实实现：调用内网版面分析 / OCR 服务（如 PaddleOCR-PPStructure）。
    - mock 实现：MockLayoutAnalyzer（下方占位，仅返回结构化骨架）。

【仅框架演示，真实部署替换】
    不写任何 OCR 安装/下载代码；仅定义接口与 mock 返回。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LayoutBlock:
    """版面分析产出的一个区块。"""
    block_type: str          # text | table | image | header_footer
    content: str
    page: int
    bbox: Optional[tuple] = None
    meta: dict = field(default_factory=dict)


class LayoutAnalyzer(ABC):
    """版面分析抽象接口：业务上层只依赖该接口。"""

    @abstractmethod
    def ocr(self, image_path: str) -> str:
        """扫描件 OCR，返回纯文本。真实实现对接内网 OCR 服务。"""
        ...

    @abstractmethod
    def analyze(self, doc_path: str) -> List[LayoutBlock]:
        """对文档做版面分析，返回有序区块列表。"""
        ...

    @abstractmethod
    def extract_tables(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """抽取表格区块（交给 table_parser 单独成块）。"""
        ...


class MockLayoutAnalyzer(LayoutAnalyzer):
    """
    模拟版面分析（仅框架演示，真实部署替换）。

    返回一份确定性结构化骨架：一个正文块 + 一个表格块 + 一个页眉块，
    便于在 examples/ 中走通后续切分 / 去重 / 索引流程。
    """

    def ocr(self, image_path: str) -> str:
        # 不真实调用 OCR；返回占位文本
        return "（OCR 占位）锅炉主蒸汽温度限额 540±5℃，超温降负荷。"

    def analyze(self, doc_path: str) -> List[LayoutBlock]:
        return [
            LayoutBlock(block_type="header_footer",
                        content="XX电厂 #1 机组运行规程 第12页", page=12),
            LayoutBlock(block_type="text",
                        content="锅炉主蒸汽温度正常运行限额为 540±5℃，超温需降负荷。",
                        page=12, meta={"section": "5.2 温度限额"}),
            LayoutBlock(block_type="table",
                        content="设备|额定|报警|跳闸\n锅炉主汽温|540|545|550\n润滑油压|0.15|0.10|0.08",
                        page=12, meta={"table_no": "表5-2"}),
        ]

    def extract_tables(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        return [b for b in blocks if b.block_type == "table"]
