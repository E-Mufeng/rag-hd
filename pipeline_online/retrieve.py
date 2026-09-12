"""
retrieve.py —— C3 检索（dense HNSW + BM25 两路，RRF 融合）

职责：对改写后的 query 做两路召回并融合：
    - dense 路：向量库（Milvus HNSW）ANN 召回；
    - BM25 路：关键词召回（OpenSearch BM25）；
    - 融合：RRF（Reciprocal Rank Fusion）倒数排名融合，
            score = Σ 1/(k + rank_i)，k 为平滑常数（configs.retrieval.rrf_k）。
所属链路：在线 C3。
对接抽象接口：VectorStore（稠密）、BM25Store（关键词），二者均提供真实/mock 两套实现。

【仅框架演示，真实部署替换】
    真实实现：
      - MilvusVectorStore：连接 http://milvus:19530（HNSW 索引）
      - OpenSearchBM25：连接 http://opensearch:9200（磁盘持久化）
    本文件用 MockVectorStore / MockBM25 走通融合逻辑，不连接任何服务。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    dense_rank: int = 0
    bm25_rank: int = 0
    rrf_score: float = 0.0
    meta: dict = field(default_factory=dict)


class VectorStore(ABC):
    """稠密向量检索抽象接口（Milvus standalone HNSW）。"""
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, str, float, dict]]:
        """返回 [(chunk_id, text, score, meta), ...]，按相关性降序。"""
        ...


class BM25Store(ABC):
    """BM25 关键词检索抽象接口（OpenSearch）。"""
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, str, float, dict]]:
        ...


def _rrf_merge(dense: List[Tuple], bm25: List[Tuple], k: int = 60) -> List[RetrievedChunk]:
    """
    RRF 倒数排名融合：
        rrf_score(chunk) = 1/(k + rank_dense) + 1/(k + rank_keyword)
    两路各自按自身排序给 rank（从 1 开始）；只在其中一路出现的，另一路 rank 缺失不计。
    """
    merged: Dict[str, RetrievedChunk] = {}
    for rank, (cid, text, _s, meta) in enumerate(dense, start=1):
        merged.setdefault(cid, RetrievedChunk(chunk_id=cid, text=text, meta=meta))
        merged[cid].dense_rank = rank
    for rank, (cid, text, _s, meta) in enumerate(bm25, start=1):
        merged.setdefault(cid, RetrievedChunk(chunk_id=cid, text=text, meta=meta))
        merged[cid].bm25_rank = rank
    for c in merged.values():
        c.rrf_score = round(
            (1 / (k + c.dense_rank) if c.dense_rank else 0)
            + (1 / (k + c.bm25_rank) if c.bm25_rank else 0), 6)
    out = list(merged.values())
    out.sort(key=lambda x: x.rrf_score, reverse=True)
    return out


def retrieve(query: str, vectorstore: VectorStore, bm25store: BM25Store,
             dense_top_k: int = 10, bm25_top_k: int = 10, rrf_k: int = 60
             ) -> List[RetrievedChunk]:
    """
    两路检索 + RRF 融合入口。
    返回融合后按 rrf_score 降序的候选列表，供下游 Rerank 精排。
    """
    dense = vectorstore.search(query, dense_top_k)
    bm25 = bm25store.search(query, bm25_top_k)
    return _rrf_merge(dense, bm25, k=rrf_k)
