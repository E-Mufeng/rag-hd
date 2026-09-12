"""
rewrite.py —— C1 问题改写（术语规范化 + HyDE 可选开关）

职责：把用户口语化/不规范的提问改写为更利于检索的标准形态。
主策略【术语规范化】：
    - 火电同义词归一（如「炉子」->「锅炉」，「汽轮机」别称统一）；
    - 实体/机组归一（「1号机」「#1机」「一号机组」->「#1机组」）；
    - 故障码标准化（「F123」全大写、补零）。
可选【HyDE】：
    - 生成一段假设性答案作为检索向量，提升召回；
    - 默认关闭（A/B 实验开关），由 configs.rewrite.hyde_enabled 控制。
所属链路：在线 C1。
对接抽象接口：HydeGenerator（抽象，默认不启用）。

【仅框架演示，真实部署替换】
    HyDE 真实实现调用内网 LLM 生成假设文档；本框架不调用模型，仅保留开关与接口。
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    used_hyde: bool = False
    hyde_doc: Optional[str] = None


# 火电同义词 / 别称归一词典（演示子集，真实部署按厂内术语库扩充）
_SYNONYMS = {
    "炉子": "锅炉", "大炉": "锅炉", "汽机": "汽轮机", "小机": "小汽轮机",
    "给水泵机": "给水泵", "主变": "主变压器", "厂变": "厂用变压器",
}
# 机组归一
_UNIT_PATTERNS = [
    ("1号机", "#1机组"), ("#1机", "#1机组"), ("一号机组", "#1机组"),
    ("2号机", "#2机组"), ("#2机", "#2机组"), ("二号机组", "#2机组"),
]


def _replace_safe(text: str, mapping: dict) -> str:
    """按词典替换，但用负向预查避免「已规范形态被再规范一次」导致叠字
    （如「汽轮机」含「汽机」、若直接 replace 会变「汽轮机机」）。
    仅当 value 以 key 开头时才需要加 lookahead 排除 value 的后续字符。"""
    for k, v in mapping.items():
        if v.startswith(k):
            la = re.escape(v[len(k):])
            text = re.sub(re.escape(k) + f"(?!{la})", v, text)
        else:
            text = text.replace(k, v)
    return text


def _normalize_terms(text: str) -> str:
    text = _replace_safe(text, _SYNONYMS)
    text = _replace_safe(text, dict(_UNIT_PATTERNS))
    # 故障码标准化：F 后数字补零到 3 位（演示）
    text = re.sub(r"\bF(\d{1,2})\b", lambda m: "F" + m.group(1).zfill(3), text)
    return text


class HydeGenerator:
    """HyDE 抽象接口（默认关闭）。真实实现调用内网 LLM 生成假设文档。"""
    def generate(self, query: str) -> str:
        raise NotImplementedError("HyDE 真实实现需对接内网 LLM；当前默认关闭")


def rewrite(query: str, use_hyde: bool = False,
            hyde: Optional[HydeGenerator] = None) -> RewriteResult:
    """
    问题改写入口：
      1) 术语规范化（主，始终执行）；
      2) HyDE（可选，use_hyde=True 且提供 hyde 实现时启用）。
    """
    rewritten = _normalize_terms(query)
    hyde_doc = None
    if use_hyde and hyde is not None:
        hyde_doc = hyde.generate(query)
    return RewriteResult(original=query, rewritten=rewritten,
                         used_hyde=use_hyde, hyde_doc=hyde_doc)
