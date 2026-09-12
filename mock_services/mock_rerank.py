"""
mock_rerank —— Cross-Encoder 精排模拟实现

职责：模拟 BGE-Reranker-v2-m3 等 cross-encoder 的(query, doc) 打分接口。
所属链路：在线 C4 Rerank 精排。
对接抽象接口：pipeline_online/rerank.py 中的 Reranker 抽象基类。

【仅框架演示，真实部署替换】
    真实实现（rerank.py 内的 CrossEncoderReranker）调用内网 rerank 服务：
        http://127.0.0.1:8003/v1/rerank （或本地加载 cross-encoder 权重）
    本 Mock 不加载权重，仅用「query 与 doc 的词重叠度 + 长度归一」近似 0~1 校准分。
"""
from __future__ import annotations

from typing import List, Tuple


def _overlap_score(query: str, doc: str) -> float:
    # 采用「查询字符在文档中的覆盖率」作为相关性代理（演示用）：
    #   越大表示文档覆盖查询信息越多。
    q = set(query)
    d = set(doc)
    if not q:
        return 0.0
    return len(q & d) / len(q)


class MockReranker:
    """
    模拟 Cross-Encoder 精排打分。

    rerank(query, candidates) -> [(chunk_id, doc_text, score), ...] 按 score 降序。
    分数近似 0~1（与真实校准分同区间，便于复用置信度阈值逻辑）。
    """

    def rerank(self, query: str, candidates: List[Tuple[str, str]],
               top_n: int = 5) -> List[Tuple[str, str, float]]:
        scored = []
        for cid, text in candidates:
            base = _overlap_score(query, text)
            # 适度放大，使强相关文档分数进入 Top1 兜底区间（>=0.92），
            # 演示「Top1 高分兜底」策略；其余按 0~1 校准分。
            score = round(min(1.0, base * 1.15), 4)
            scored.append((cid, text, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]
