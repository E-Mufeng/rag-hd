"""
router.py —— 离线文档分发路由

职责：根据文件类型把文档分发到不同的解析路径：
    - PDF（文本型）  -> 直接版面分析
    - Word(docx)     -> 直接版面分析
    - 扫描件(图片/扫描PDF) -> 先 OCR（版面分析抽象接口内的 OCR 适配器）再版面分析
    - 其他           -> 拒收并记日志

所属链路：离线 A1。
对接抽象接口：layout_analyzer.LayoutAnalyzer（含 OCR 适配器抽象）。

【仅框架演示，真实部署替换】
    OCR 适配器真实实现调用内网 PaddleOCR / 商业 OCR 服务；本框架不写任何 OCR
    安装/下载脚本，仅保留接口与 mock 返回。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    PDF_TEXT = "pdf_text"
    WORD = "word"
    SCAN = "scan"      # 扫描件 / 图片
    UNKNOWN = "unknown"


@dataclass
class RoutedDoc:
    path: str
    doc_type: DocType
    need_ocr: bool


# 扩展名 -> 类型初判
_EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
}


def detect_type(path: str) -> DocType:
    """按扩展名初判类型；扫描件需进一步由版面分析层判断是否含文本层。"""
    ext = os.path.splitext(path)[-1].lower()
    kind = _EXT_MAP.get(ext, "unknown")
    if kind == "pdf":
        # 演示逻辑：pdf 默认按文本型路由；真实实现可先探测是否有文本层
        return DocType.PDF_TEXT
    if kind == "word":
        return DocType.WORD
    if kind == "image":
        return DocType.SCAN
    return DocType.UNKNOWN


def route(path: str) -> RoutedDoc:
    """
    文档分发路由：返回类型判定与是否需要 OCR。
    业务上层依据 RoutedDoc 选择后续解析链路。
    """
    doc_type = detect_type(path)
    need_ocr = doc_type == DocType.SCAN
    return RoutedDoc(path=path, doc_type=doc_type, need_ocr=need_ocr)


# 真实部署时，OCR 适配器的抽象接口见 layout_analyzer.LayoutAnalyzer.ocr()，
# 这里仅演示路由决策，不执行任何 OCR 调用。
