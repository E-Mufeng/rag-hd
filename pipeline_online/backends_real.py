"""
backends_real.py —— 真实后端实现（对接内网服务 / 中间件）

【定位】本文件是「生产部署」侧代码，与 mock_services/* 一一对应，业务上层在
        config 指定 backend != mock 时被 factory.build_backends() 选用。

【核心约束】
    1) 所有类均通过 HTTP / 内网服务 / 中间件客户端对接，**不下载任何模型权重**、
       不拉起任何服务（本仓库定位为架构框架演示，真实运行在内网服务就绪后进行）；
    2) 重依赖（pymilvus / opensearchpy / openai / requests）全部「方法内懒加载」，
       因此本模块可被安全 import，mock 演示不会因缺包而崩；
    3) 本地没有对应中间件/服务时，方法调用会抛连接错误——这是预期的，不是 bug。

【与 pipeline 模块的对应关系】
    在线（C 段）
    MilvusVectorStore  -> retrieve.VectorStore   （dense HNSW 一路）
    OpenSearchBM25     -> retrieve.BM25Store     （BM25 一路）
    VLLMClient         -> generate.LLMClient     （C5 生成）
    BGERerankerService -> rerank.Reranker        （C4 精排）
    MiniLMIntentService-> intent.IntentModel     （C2 意图）
    离线（A 段）
    LayoutAnalyzerService   -> layout_analyzer.LayoutAnalyzer   （A2 版面 / OCR）
    MilvusVectorIndexWriter -> incremental_index.VectorIndexWriter（A9 增量索引）
"""
from __future__ import annotations

from abc import ABC
from typing import Dict, List, Tuple

from pipeline_online.retrieve import VectorStore, BM25Store
from pipeline_online.rerank import Reranker
from pipeline_online.generate import LLMClient
from pipeline_online.intent import IntentModel, IntentDecision
from pipeline_offline.layout_analyzer import LayoutAnalyzer, LayoutBlock
from pipeline_offline.incremental_index import VectorIndexWriter


# ============================================================================
# C3-dense 一路：Milvus standalone HNSW
# ============================================================================
class MilvusVectorStore(VectorStore):
    """稠密向量检索真实实现：对接内网 Milvus standalone（HNSW 索引）。

    query 文本需先用内网 embedding 服务向量化（见 pipeline_offline/embedder.py
    的 EmbeddingService），本类不直接做向量化。
    """

    def __init__(self, uri: str, collection: str, dim: int = 1024,
                 index_type: str = "HNSW", metric_type: str = "L2"):
        self.uri = uri
        self.collection = collection
        self.dim = dim
        self.index_type = index_type
        self.metric_type = metric_type
        self._client = None

    def _connect(self):
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=self.uri)
        return self._client

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float, dict]]:
        from pipeline_offline.embedder import EmbeddingService
        qvec = EmbeddingService().embed([query])[0]
        client = self._connect()
        res = client.search(
            collection_name=self.collection,
            data=[qvec],
            limit=top_k,
            search_params={"params": {"ef": 64}},
        )
        out = []
        for hit in res[0]:
            entity = hit.get("entity", {})
            out.append((
                str(hit.get("id")),
                entity.get("text", ""),
                float(hit.get("distance", 0.0)),
                entity.get("meta", {}) or {},
            ))
        return out


# ============================================================================
# C3-BM25 一路：OpenSearch 容器
# ============================================================================
class OpenSearchBM25(BM25Store):
    """BM25 真实实现：对接内网 OpenSearch 容器。

    - 索引磁盘持久化，服务重启不丢索引；
    - 文档写入见 offline 管线的增量索引模块（文本同步写 OpenSearch）；
    - 查询用 OpenSearch 的 BM25（best_fields multi_match），中文建议配 IK 分词插件。
    """

    def __init__(self, uri: str = "http://opensearch:9200", index: str = "huodian_bm25",
                 user: str = "", password: str = ""):
        self.uri = uri
        self.index = index
        self._auth = (user, password) if user else None
        self._client = None

    def _connect(self):
        if self._client is None:
            from opensearchpy import OpenSearch
            self._client = OpenSearch(hosts=[self.uri], http_auth=self._auth,
                                      use_ssl=False, verify_certs=False)
        return self._client

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float, dict]]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^1.0", "title^1.5"],
                    "type": "best_fields",
                }
            },
        }
        res = self._connect().search(index=self.index, body=body)
        out = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            out.append((h["_id"], src.get("text", ""), float(h["_score"] or 0.0),
                        src.get("meta", {}) or {}))
        return out


# ============================================================================
# C5 生成：vLLM（Qwen3-14B，OpenAI 兼容）
# ============================================================================
class VLLMClient(LLMClient):
    """生成真实实现：调用内网 vLLM OpenAI 兼容服务。

    数据不出域：base_url 指向厂内 vLLM 服务，model 为已载入的 Qwen3-14B。
    Qwen3 属混合推理（thinking / non-thinking）模型，火电运维问答以低延迟
    为第一优先，故默认关闭思维链（enable_thinking=False）；对少数需要多步
    推理的工况分析类问题，可按请求临时打开。
    本类不写 model.from_pretrained / 自动下载代码，模型由内网推理服务托管。
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "Qwen3-14B", temperature: float = 0.1,
                 enable_thinking: bool = False):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def generate(self, prompt: str, contexts: List[str], temperature: float = 0.1) -> str:
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=800,
            # vLLM 侧透传 chat template 参数，关闭 Qwen3 思维链
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )
        return resp.choices[0].message.content or ""


# ============================================================================
# C4 精排：BGE-Reranker-v2-m3（cross-encoder，0~1 校准分）
# ============================================================================
class BGERerankerService(Reranker):
    """Cross-Encoder 精排真实实现：调用内网 rerank 服务。

    服务形态（按内网现状二选一）：
      A) FlagEmbedding 起的 rerank 服务（POST query + documents -> 归一化分数）；
      B) vLLM 以 cross-encoder 模型暴露的 /v1/rerank。
    返回 0~1 校准分，供 rerank.aggregate_confidence 做阈值分级。
    """

    def __init__(self, service_url: str = "http://127.0.0.1:8003/v1/rank"):
        self.service_url = service_url

    def rerank(self, query: str, candidates: List[Tuple[str, str]],
               top_n: int = 5) -> List[Tuple[str, str, float]]:
        import requests
        docs = [text for _, text in candidates]
        resp = requests.post(self.service_url,
                             json={"query": query, "documents": docs},
                             timeout=60)
        resp.raise_for_status()
        scores = resp.json().get("scores") or resp.json().get("results")
        scored = [(cid, text, round(float(s), 4))
                  for (cid, text), s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]


# ============================================================================
# C2 意图：MiniLM 轻模型（内网 embedding 服务的句向量 + 最近邻分类）
# ============================================================================
class MiniLMIntentService(IntentModel):
    """意图真实实现：调用内网 MiniLM embedding 服务得到句向量，再用「模板最近邻」分类。

    MiniLM 句向量 + 4 类模板余弦即可，CPU、毫秒级、零幻觉风险；
    分类头用「最近邻规则」实现，无需训练、易解释。模板向量首次调用时缓存。
    """

    def __init__(self, service_url: str = "http://127.0.0.1:8002/v1/embeddings",
                 templates: Dict[str, str] = None):
        self.service_url = service_url
        self.templates = templates or {
            "procedure": "火电运行操作规程、参数限额、启停操作、系统操作方法",
            "fault": "火电设备故障、异常、事故处理、泄漏跳闸振动超温停机",
            "chitchat": "你好谢谢打招呼闲聊寒暄",
            "out_of_domain": "与火电运维无关的问题如足球股票菜谱旅游",
        }
        self._tmpl_vecs = None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        resp = requests.post(self.service_url, json={"input": texts}, timeout=30)
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    def _ensure_tmpl(self):
        if self._tmpl_vecs is None:
            keys = list(self.templates.keys())
            vecs = self._embed([self.templates[k] for k in keys])
            self._tmpl_vecs = (keys, vecs)

    @staticmethod
    def _cos(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def classify(self, text: str) -> IntentDecision:
        self._ensure_tmpl()
        keys, vecs = self._tmpl_vecs
        qv = self._embed([text])[0]
        sims = [self._cos(qv, v) for v in vecs]
        best = max(range(len(sims)), key=lambda i: sims[i])
        return IntentDecision(label=keys[best], confidence=round(float(sims[best]), 4),
                              passed_pre_gate=False, reason="minilm_service")


# ============================================================================
# A2 版面分析 / OCR：内网版面解析服务
# ============================================================================
class LayoutAnalyzerService(LayoutAnalyzer):
    """版面分析真实实现：调用内网版面解析 / OCR 服务。

    服务形态（按内网现状二选一）：
      A) PaddleOCR-PPStructure 封装成的 HTTP 服务；
      B) 商业 OCR / 版面还原 SDK 的网关。
    约定返回：{"blocks": [{"type": "text|table|image|header_footer",
                          "content": "...", "page": 12, "bbox": [x0,y0,x1,y1]}]}
    扫描件先走 ocr()，再由 analyze() 还原版面；文本型 PDF / Word 直接 analyze()。
    """

    def __init__(self, service_url: str = "http://127.0.0.1:8004/v1/layout"):
        self.service_url = service_url

    def ocr(self, image_path: str) -> str:
        import requests
        resp = requests.post(self.service_url.rstrip("/") + "/ocr",
                             json={"image_path": image_path}, timeout=120)
        resp.raise_for_status()
        return resp.json().get("text", "")

    def analyze(self, doc_path: str) -> List[LayoutBlock]:
        import requests
        resp = requests.post(self.service_url.rstrip("/") + "/analyze",
                             json={"doc_path": doc_path}, timeout=300)
        resp.raise_for_status()
        blocks: List[LayoutBlock] = []
        for b in resp.json().get("blocks", []):
            blocks.append(LayoutBlock(
                block_type=b.get("type", "text"),
                content=b.get("content", ""),
                page=int(b.get("page", 0)),
                bbox=tuple(b["bbox"]) if b.get("bbox") else None,
                meta=b.get("meta", {}) or {},
            ))
        return blocks

    def extract_tables(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        return [b for b in blocks if b.block_type == "table"]


# ============================================================================
# A9 增量索引写入：Milvus(HNSW) + OpenSearch 双写
# ============================================================================
class MilvusVectorIndexWriter(VectorIndexWriter):
    """增量索引真实实现：指纹去重后双写向量库与 BM25 索引。

    写入约定（与 pipeline_offline/incremental_index.py 的指纹策略一致）：
      - 以 content_hash + version 作为幂等依据，同指纹不重复写；
      - 文档版本更新时，先按 document 删除旧 chunk，再写入新 chunk，
        避免旧规程残留在库里被检索到（这是运维知识库最容易出的事故）；
      - 文本同步写入 OpenSearch，保证 BM25 一路与向量库口径一致。
    """

    def __init__(self, milvus_uri: str, collection: str, dim: int = 1024,
                 os_uri: str = "http://opensearch:9200", os_index: str = "huodian_bm25"):
        self.milvus_uri = milvus_uri
        self.collection = collection
        self.dim = dim
        self.os_uri = os_uri
        self.os_index = os_index
        self._milvus = None
        self._os = None

    def _clients(self):
        if self._milvus is None:
            from pymilvus import MilvusClient
            self._milvus = MilvusClient(uri=self.milvus_uri)
        if self._os is None:
            from opensearchpy import OpenSearch
            self._os = OpenSearch(hosts=[self.os_uri], use_ssl=False, verify_certs=False)
        return self._milvus, self._os

    def upsert(self, chunks: List[dict]) -> List[str]:
        from pipeline_offline.embedder import EmbeddingService
        milvus, os_client = self._clients()
        embedder = EmbeddingService()

        # 同一文档旧版本先删（按 document 维度清）
        docs = {c["document"] for c in chunks}
        for doc in docs:
            self._delete_by_document(doc)

        vecs = embedder.embed([c["text"] for c in chunks]) if chunks else []
        written: List[str] = []
        rows = []
        for c, v in zip(chunks, vecs):
            rows.append({"id": c["chunk_id"], "vector": v, "text": c["text"],
                         "meta": {"document": c["document"], "version": c["version"]}})
            written.append(c["chunk_id"])
        if rows:
            milvus.upsert(collection_name=self.collection, data=rows)
            for c in chunks:
                os_client.index(index=self.os_index, id=c["chunk_id"],
                                body={"text": c["text"], "title": c["document"],
                                      "meta": {"document": c["document"], "version": c["version"]}})
        return written

    def _delete_by_document(self, document: str) -> None:
        milvus, os_client = self._clients()
        try:
            milvus.delete(collection_name=self.collection,
                          filter=f'meta["document"] == "{document}"')
            os_client.delete_by_query(index=self.os_index,
                                      body={"query": {"term": {"meta.document": document}}})
        except Exception:
            # 首次入库该文档尚不存在，忽略即可；真实运行需在此处记 WARN 日志
            pass

    def delete_by_fingerprint(self, fingerprint: str) -> None:
        milvus, _ = self._clients()
        milvus.delete(collection_name=self.collection, filter=f'id == "{fingerprint}"')
