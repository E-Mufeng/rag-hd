"""
dedup.py —— 两级去重（MinHash 粗筛 + embedding 余弦精筛，阈值分档）

职责：离线入湖前对 chunk 去重，抑制多源文档重复段落带来的索引膨胀与检索噪声。
两级策略：
    L0 粗筛：MinHash 估计 Jaccard 相似度，>= minhash_threshold 才进入 L1；
    L1 精筛：embedding 余弦相似度分档：
        >= embedding_high   -> 自动合并（保留高优先级 chunk，删另一份）
        [embedding_medium, embedding_high) -> 待人工复核（标记，不自动删）
        <  embedding_medium -> 视为不重复，保留
所属链路：离线 A8。
对接抽象接口：需要 embedding 服务（抽象，见下方 Embedding 抽象）。

【仅框架演示，真实部署替换】
    MinHash 可用 datasketch；embedding 余弦调用内网 embedding 服务。
    本框架用确定性 mock embedding（hash 向量）演示分档逻辑，不下载模型。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from pipeline_offline.semantic_chunker import Chunk
from pipeline_offline.embedder import Embedder, MockEmbedder


class DedupTier(str, Enum):
    KEEP_BOTH = "keep_both"          # 不重复
    AUTO_MERGE = "auto_merge"        # 自动合并
    NEED_REVIEW = "need_review"      # 待人工复核


@dataclass
class DedupDecision:
    chunk_id: str
    twin_id: str
    tier: DedupTier
    minhash_sim: float
    cosine: float


# —— embedding 抽象（真实部署对接内网服务）——
def mock_embed(text: str, dim: int = 16) -> List[float]:
    """确定性 mock embedding（hash 向量，仅演示）。"""
    vec = []
    for i in range(dim):
        h = hashlib.md5(f"{text}#{i}".encode()).digest()
        vec.append((int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2 - 1)
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _minhash_jaccard(a: str, b: str, n_perm: int = 32) -> float:
    """简化 MinHash：比较 n_perm 个哈希最小值的重合比例，近似 Jaccard。"""
    def sig(t: str):
        shingles = set(t[i:i + 3] for i in range(len(t) - 2))
        return [min(hashlib.md5((s + f"#{p}").encode()).digest()[:4]
                    for s in shingles) for p in range(n_perm)] if shingles else [b"0"] * n_perm
    sa, sb = sig(a), sig(b)
    return sum(1 for x, y in zip(sa, sb) if x == y) / max(1, n_perm)


class TwoStageDedup:
    """
    两级去重器：对候选 chunk 两两比较，输出去重决策。

    用法：decisions = TwoStageDedup(th).run(chunks)
    业务上层依据 tier 处理：auto_merge 删低优先级副本，need_review 入人工队列。
    """

    def __init__(self, minhash_threshold: float = 0.85,
                 embedding_high: float = 0.95, embedding_medium: float = 0.85,
                 embedder: Embedder = None, dim: int = 16):
        self.minhash_threshold = minhash_threshold
        self.embedding_high = embedding_high
        self.embedding_medium = embedding_medium
        self.dim = dim
        # 真实部署注入 EmbeddingService（内网 BGE-M3）；默认 MockEmbedder（确定性哈希，离线演示）
        self._embedder = embedder or MockEmbedder(dim=dim)
        self._cache: dict = {}

    def _emb(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self._embedder.embed([text])[0]
        return self._cache[text]

    def compare(self, ca: Chunk, cb: Chunk) -> DedupDecision:
        mh = _minhash_jaccard(ca.text, cb.text)
        if mh < self.minhash_threshold:
            return DedupDecision(ca.chunk_id, cb.chunk_id, DedupTier.KEEP_BOTH, mh, 0.0)
        cos = _cosine(self._emb(ca.text), self._emb(cb.text))
        if cos >= self.embedding_high:
            tier = DedupTier.AUTO_MERGE
        elif cos >= self.embedding_medium:
            tier = DedupTier.NEED_REVIEW
        else:
            tier = DedupTier.KEEP_BOTH
        return DedupDecision(ca.chunk_id, cb.chunk_id, tier, mh, round(cos, 4))

    def run(self, chunks: List[Chunk]) -> List[DedupDecision]:
        decisions: List[DedupDecision] = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                decisions.append(self.compare(chunks[i], chunks[j]))
        return decisions
