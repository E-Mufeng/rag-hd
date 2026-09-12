"""
rerank.py —— C4 精排（CrossEncoder 抽象 + 置信度聚合 + 阈值分级）

职责：对检索融合后的候选做 cross-encoder 精排打分，并聚合出「整体置信度」，
      作为【检索后置信度闸门】（与意图前置闸门相互独立）。
置信度聚合策略：
    - Top-3 加权聚合：conf = w1*s1 + w2*s2 + w3*s3（权重递减，突出头部）；
    - Top1 高分兜底：若 Top1 分数极高（>= top1_fallback，如 0.92），直接采用 Top1 分，
      避免被后面低分拉低（典型「强单点命中」场景，如明确参数限额查询）。
阈值分级（双闸门之「置信度闸门」）：
    - conf >= 0.70  -> high：直接作答 + 溯源；
    - 0.40 <= conf < 0.70 -> medium：给答案 + 标注低置信提示 + 建议核实；
    - conf < 0.40   -> low：拒答（落盘低置信/拒答日志）。
所属链路：在线 C4。
对接抽象接口：Reranker（业务上层只依赖它）。

【仅框架演示，真实部署替换】
    CrossEncoderReranker 真实实现调用内网 rerank 服务：
        http://127.0.0.1:8003/v1/rerank （或本地 cross-encoder 权重）。
    本文件用 MockReranker 演示聚合与分级逻辑，不加载模型。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple


@dataclass
class RerankResult:
    chunk_id: str
    text: str
    score: float


@dataclass
class ConfidenceDecision:
    level: str              # high | medium | low(reject)
    confidence: float
    top_results: List[RerankResult]
    answer_hint: str        # 给生成模块的策略提示


# Top-3 加权（突出头部）
_TOP3_WEIGHTS = (0.5, 0.3, 0.2)
_TOP1_FALLBACK = 0.92       # Top1 高分兜底阈值
_HIGH, _LOW = 0.70, 0.40


class Reranker(ABC):
    """Cross-Encoder 精排抽象接口。"""
    @abstractmethod
    def rerank(self, query: str, candidates: List[Tuple[str, str]],
               top_n: int = 5) -> List[Tuple[str, str, float]]:
        ...


def aggregate_confidence(scores: List[float]) -> float:
    """
    置信度聚合：Top-3 加权 + Top1 高分兜底。
    scores 已按降序排列。
    """
    if not scores:
        return 0.0
    top3 = scores[:3]
    # Top1 高分兜底：单点强命中直接采用
    if top3[0] >= _TOP1_FALLBACK:
        return round(top3[0], 4)
    # Top-3 加权聚合（不足 3 个时按实际数量归一权重）
    w = _TOP3_WEIGHTS[:len(top3)]
    w = tuple(x / sum(w) for x in w)
    conf = sum(wi * si for wi, si in zip(w, top3))
    return round(conf, 4)


def _classify(conf: float) -> ConfidenceDecision:
    if conf >= _HIGH:
        return ConfidenceDecision("high", conf, [], "direct_answer")
    if conf >= _LOW:
        return ConfidenceDecision("medium", conf, [], "low_conf_hint")
    return ConfidenceDecision("low", conf, [], "reject")


def log_low_conf(query: str, conf: float, log_dir: str = "./data/logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "low_conf.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query, "conf": conf,
        }, ensure_ascii=False) + "\n")


def rerank(query: str, candidates: List[Tuple[str, str]], reranker: Reranker,
           top_n: int = 5, log_dir: str = "./data/logs") -> ConfidenceDecision:
    """
    精排 + 置信度聚合 + 阈值分级（检索后置信度闸门）。

    candidates: [(chunk_id, text), ...]（来自 retrieve.retrieve 的融合结果）。
    返回 ConfidenceDecision，业务上层据此决定：直接作答 / 低置信提示 / 拒答。
    """
    scored = reranker.rerank(query, candidates, top_n=top_n)
    results = [RerankResult(cid, text, score) for cid, text, score in scored]
    scores = [s for _, _, s in scored]
    conf = aggregate_confidence(scores)
    decision = _classify(conf)
    decision.top_results = results
    if decision.level == "low":
        log_low_conf(query, conf, log_dir)
    return decision
