"""
incremental_index.py —— 增量索引（content + version 哈希，删旧 chunk）

职责：文档更新时，以 (content_hash + version_hash) 作为 chunk 唯一指纹，
      决定「新增 / 复用 / 删除旧版本」：
        - 指纹已存在 -> 跳过（复用）；
        - content 变而 version 变 -> 写入新 chunk 并删除同 document 旧指纹 chunk；
        - 旧文档中不再出现的指纹 -> 删除。
所属链路：离线 A9。
对接抽象接口：VectorIndexWriter（抽象）；本文件不直接连接 Milvus。

【仅框架演示，真实部署替换】
    真实实现（MilvusVectorIndexWriter）会：
      - 用 content_hash+version 作为主键 upsert 到 Milvus(HNSW)；
      - 同时把文本写入 OpenSearch 的 BM25 索引。
    本框架只维护一个内存指纹表，演示「删旧 chunk」的业务逻辑，不真实调库。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IndexRecord:
    chunk_id: str
    fingerprint: str      # content_hash + version_hash
    document: str
    version: str


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


class VectorIndexWriter:
    """向量索引写入抽象接口（业务上层只依赖它）。"""

    def upsert(self, records: List[IndexRecord]) -> None:
        raise NotImplementedError

    def delete_by_fingerprint(self, fingerprint: str) -> None:
        raise NotImplementedError


class MockIncrementalIndexer(VectorIndexWriter):
    """
    模拟增量索引（仅框架演示，真实部署替换为 Milvus 实现）。

    维护内存指纹表，体现「增量写入 + 删旧 chunk」逻辑：
      - 同指纹复用；
      - 同 document 不同指纹 -> 删旧写新；
      - 某 document 本轮未出现的旧指纹 -> 删除（文档被改小/段落移除）。
    """

    def __init__(self):
        self._store: Dict[str, IndexRecord] = {}      # fingerprint -> record
        self._doc_chunks: Dict[str, set] = {}          # document -> {fingerprint}

    def _fp(self, content: str, version: str) -> str:
        return _hash(content) + _hash(version)

    def upsert(self, chunks: List[dict]) -> List[str]:
        """
        chunks: [{"chunk_id","text","document","version"}]
        返回本次实际新增/更新的 chunk_id 列表（供上层回写 BM25 索引）。
        """
        written: List[str] = []
        for c in chunks:
            fp = self._fp(c["text"], c["version"])
            doc = c["document"]
            self._doc_chunks.setdefault(doc, set())
            # 删旧：同 document 下、本轮未携带的旧指纹
            for old_fp in list(self._doc_chunks[doc]):
                if old_fp != fp and old_fp not in {self._fp(x["text"], x["version"]) for x in chunks}:
                    self.delete_by_fingerprint(old_fp)
            if fp not in self._store:
                self._store[fp] = IndexRecord(c["chunk_id"], fp, doc, c["version"])
                self._doc_chunks[doc].add(fp)
                written.append(c["chunk_id"])
        return written

    def delete_by_fingerprint(self, fingerprint: str) -> None:
        rec = self._store.pop(fingerprint, None)
        if rec:
            self._doc_chunks.get(rec.document, set()).discard(fingerprint)

    def snapshot(self) -> List[IndexRecord]:
        return list(self._store.values())
