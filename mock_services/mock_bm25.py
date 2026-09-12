"""
mock_bm25 —— BM25 关键词检索模拟实现

职责：模拟 OpenSearch BM25 检索接口（docker 模式）。
所属链路：在线检索 C3（BM25 一路）。
对接抽象接口：pipeline_online/retrieve.py 中的 BM25Store 抽象基类。

【仅框架演示，真实部署替换】
    真实实现（retrieve.py 内的 OpenSearchBM25 / RankBM25Backend）会：
      - docker：连接 http://opensearch:9200 （索引磁盘持久化，重启不丢）
    本 Mock 用 jieba 分词 + 词频统计近似 BM25 打分，仅演示流程。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    score: float
    meta: dict = field(default_factory=dict)


# 模拟语料（与 mock_vectorstore 同源，便于演示两路融合）
_CORPUS = [
    ("c001", "锅炉主蒸汽温度正常运行限额为 540±5℃，超温需降负荷。",
     {"page": 12, "equip": "锅炉", "src": "运行规程"}),
    ("c002", "汽轮机润滑油压力低报警值为 0.08 MPa，联锁启动备用泵。",
     {"page": 33, "equip": "汽轮机", "src": "运行规程"}),
    ("c003", "给水泵跳闸处理：立即抢合备用泵，失败则降负荷停机。",
     {"page": 51, "equip": "给水泵", "src": "事故处理"}),
    ("c004", "脱硫吸收塔浆液 pH 应控制在 5.2~5.8，偏低加石灰石浆液。",
     {"page": 67, "equip": "脱硫", "src": "运行规程"}),
    ("c005", "发电机定子冷却水流量低低（< 45 t/h）触发停机保护。",
     {"page": 88, "equip": "发电机", "src": "保护定值"}),
]


class MockBM25:
    """
    模拟 BM25 检索（OpenSearch 抽象）。

    search() 行为与真实 BM25Store.search() 一致：输入 query -> 返回 Top-K 命中。
    这里用「查询词在文档中出现次数 / 文档长度」的简化 BM25 近似。
    """

    def __init__(self, index: str = "huodian_bm25"):
        self.index = index
        self._docs = [{"id": i, "text": t, "meta": m} for i, t, m in _CORPUS]
        self._avg_len = sum(len(d["text"]) for d in self._docs) / max(1, len(self._docs))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 简化分词：按字符切（演示用）；真实实现用 jieba / IK 分词
        return list(text)

    def search(self, query: str, top_k: int = 5):
        q_tokens = self._tokenize(query)
        hits = []
        for d in self._docs:
            doc_tokens = self._tokenize(d["text"])
            tf = sum(1 for t in doc_tokens if t in set(q_tokens))
            if tf == 0:
                continue
            # 简化 BM25：tf * idf，idf 用常数近似
            idf = math.log(1 + (len(self._docs) - 1) / 1)
            dl = max(1, len(doc_tokens))
            score = idf * (tf * (1.2 + 1)) / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / self._avg_len))
            # 返回值对齐 BM25Store 抽象接口：Tuple[chunk_id, text, score, meta]
            hits.append((d["id"], d["text"], round(score, 4), d["meta"]))
        hits.sort(key=lambda h: h[2], reverse=True)
        return hits[:top_k]
