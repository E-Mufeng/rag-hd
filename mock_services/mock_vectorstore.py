"""
mock_vectorstore —— 向量库（dense / HNSW）模拟实现

职责：模拟 Milvus standalone（HNSW）的稠密向量检索接口。
所属链路：在线检索 C3（dense 一路）。
对接抽象接口：pipeline_online/retrieve.py 中的 VectorStore 抽象基类。

【仅框架演示，真实部署替换】
    真实实现（retrieve.py 内的 MilvusVectorStore）会连接：
      - docker 模式：http://milvus:19530 （Milvus standalone, HNSW 索引）
    本 Mock 不连接任何服务，仅用确定性哈希生成向量、用简单相似度返回候选。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockDoc:
    """模拟检索命中的文档片段。"""
    chunk_id: str
    text: str
    score: float
    meta: dict = field(default_factory=dict)


def _hash_vec(text: str, dim: int = 16) -> List[float]:
    """用文本哈希确定性生成一个低维模拟向量（仅演示，无语义）。"""
    vec = []
    for i in range(dim):
        h = hashlib.md5(f"{text}#{i}".encode("utf-8")).digest()
        # 映射到 [-1, 1]
        vec.append((int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2 - 1)
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MockVectorStore:
    """
    模拟稠密向量检索（HNSW 抽象）。

    search() 行为与真实 MilvusVectorStore.search() 一致：
        输入 query 向量 -> 返回 Top-K 候选片段及其距离分。
    这里用关键词重叠度近似「语义相似度」，便于静态阅读流程。
    """

    def __init__(self, dim: int = 16, collection: str = "huodian_chunks"):
        self.dim = dim
        self.collection = collection
        # 预置一些模拟知识库片段（火电运维场景）
        self._corpus = [
            MockDoc("c001", "锅炉主蒸汽温度正常运行限额为 540±5℃，超温需降负荷。", 0.0,
                    {"page": 12, "equip": "锅炉", "src": "运行规程"}),
            MockDoc("c002", "汽轮机润滑油压力低报警值为 0.08 MPa，联锁启动备用泵。", 0.0,
                    {"page": 33, "equip": "汽轮机", "src": "运行规程"}),
            MockDoc("c003", "给水泵跳闸处理：立即抢合备用泵，失败则降负荷停机。", 0.0,
                    {"page": 51, "equip": "给水泵", "src": "事故处理"}),
            MockDoc("c004", "脱硫吸收塔浆液 pH 应控制在 5.2~5.8，偏低加石灰石浆液。", 0.0,
                    {"page": 67, "equip": "脱硫", "src": "运行规程"}),
            MockDoc("c005", "发电机定子冷却水流量低低（< 45 t/h）触发停机保护。", 0.0,
                    {"page": 88, "equip": "发电机", "src": "保护定值"}),
        ]

    def embed(self, text: str) -> List[float]:
        """模拟 embedding（真实部署调用内网 embedding 服务）。"""
        return _hash_vec(text, self.dim)

    def search(self, query: str, top_k: int = 5):
        """返回与 query 关键词重叠度最高的 top_k 片段（模拟 HNSW ANN 召回）。

        返回值对齐 VectorStore 抽象接口：
            List[Tuple[chunk_id, text, score, meta]]
        """
        q_tokens = set(query)
        scored = []
        for doc in self._corpus:
            overlap = len(q_tokens & set(doc.text)) / max(1, len(q_tokens))
            # 用重叠度近似相似度分（0~1，越大越相关）
            score = round(overlap, 4)
            scored.append((doc.chunk_id, doc.text, score, doc.meta))
        scored.sort(key=lambda d: d[2], reverse=True)
        return scored[:top_k]
