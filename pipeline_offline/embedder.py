"""
embedder.py —— 向量化抽象（离线管线 A8 两级去重 L1 / 建库共用）

职责：把「用什么模型向量化」与业务管线解耦。
      - 离线两级去重的 L1 精筛（embedding 余弦）需要向量；
      - 增量建库时 chunk 向量化也需要向量。
提供两套实现：
      - MockEmbedder      ：【离线演示】确定性哈希向量，无语义、无下载、无 GPU；
      - EmbeddingService  ：【生产】调用内网 embedding 服务（BGE-M3 稠密 1024），
                            通过 HTTP 拿到向量，【不加载/不下载本地模型】。

【仅框架演示，真实部署替换】
    EmbeddingService.service_url 指向内网 BGE-M3 服务（如 http://127.0.0.1:8001/v1/embeddings），
    返回 dense 1024 维向量。本文件不写任何模型下载代码。
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """向量化抽象接口。业务上层（dedup / 建库）只依赖它。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回与 texts 同序的向量列表。"""
        ...

    def dim(self) -> int:
        return 1024


class MockEmbedder(Embedder):
    """【离线演示】确定性哈希向量（16 维，同文本同向量）。

    无语义、无下载、无 GPU，仅在无算力环境演示检索/去重链路；
    真实语义向量由 EmbeddingService 提供。
    """

    def __init__(self, dim: int = 16):
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            vec = []
            for i in range(self._dim):
                h = hashlib.md5(f"{t}#{i}".encode("utf-8")).digest()
                vec.append((int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2 - 1)
            out.append(vec)
        return out


class EmbeddingService(Embedder):
    """【生产】调用内网 embedding 服务（BGE-M3 dense 1024）。

    - 通过 HTTP 拿向量，不加载本地模型、不下载权重；
    - 内网服务就绪后，离线管线的去重/建库直接复用真实语义向量。
    """

    def __init__(self, service_url: str = "http://127.0.0.1:8001/v1/embeddings",
                 dim: int = 1024):
        self.service_url = service_url
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests

        resp = requests.post(self.service_url, json={"input": texts}, timeout=60)
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]
