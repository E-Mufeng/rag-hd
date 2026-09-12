"""
factory.py —— 后端装配工厂（mock / real 一键切换）

核心思想：业务代码只依赖抽象接口与 factory 返回的实例，不直接 new 具体中间件客户端。
        部署形态固定为 docker（Milvus + OpenSearch），仅通过 run.use_mock 切换：

    run.use_mock = true  -> 全部 mock（静态阅读/离线演示，不连任何服务）
    run.use_mock = false -> 真实后端：MilvusVectorStore(HNSW) + OpenSearchBM25

=> 业务代码零改动即可在「离线演示」与「内网试点」间切换，体现业务与基础设施解耦。

【注意】真实实例仅持有连接参数（uri/url），方法被调用时才真正连服务；
        本机无服务时调用会抛连接错误，属预期（框架演示不改真实运行）。
"""
from __future__ import annotations

from typing import Dict


def build_backends(cfg: Dict, use_mock: bool = None) -> Dict:
    """按配置装配后端实例字典。

    返回键：vectorstore / bm25 / reranker / llm / intent / embedder
            + layout_analyzer / index_writer（离线入湖接口使用）
    各值均为对应抽象接口的实例。
    """
    if use_mock is None:
        use_mock = bool(cfg.get("run", {}).get("use_mock", True))
    models = cfg.get("models", {})
    storage = cfg.get("storage", {})

    # ---------------- mock 模式：不连任何服务 ----------------
    if use_mock:
        from mock_services.mock_vectorstore import MockVectorStore
        from mock_services.mock_bm25 import MockBM25
        from mock_services.mock_rerank import MockReranker
        from mock_services.mock_llm import MockLLM
        from mock_services.mock_intent import MockIntentModel
        from pipeline_offline.embedder import MockEmbedder
        from pipeline_offline.layout_analyzer import MockLayoutAnalyzer
        from pipeline_offline.incremental_index import MockIncrementalIndexer
        return {
            "vectorstore": MockVectorStore(),
            "bm25": MockBM25(),
            "reranker": MockReranker(),
            "llm": MockLLM(),
            "intent": MockIntentModel(),
            "embedder": MockEmbedder(),
            "layout_analyzer": MockLayoutAnalyzer(),
            "index_writer": MockIncrementalIndexer(),
        }

    # ---------------- 真实模式：docker 模式固定对接 Milvus + OpenSearch ----------------
    from pipeline_online.backends_real import (
        MilvusVectorStore, OpenSearchBM25,
        VLLMClient, BGERerankerService, MiniLMIntentService,
        LayoutAnalyzerService, MilvusVectorIndexWriter,
    )
    from pipeline_offline.embedder import EmbeddingService

    vs = storage.get("vectorstore", {})
    bm = storage.get("bm25", {})

    vectorstore = MilvusVectorStore(
        uri=vs.get("uri"),
        collection=vs.get("collection"),
        index_type=vs.get("index_type", "HNSW"),
        metric_type=vs.get("metric_type", "L2"),
    )
    bm25 = OpenSearchBM25(uri=bm.get("uri"), index=bm.get("index"))

    return {
        "vectorstore": vectorstore,
        "bm25": bm25,
        "reranker": BGERerankerService(models.get("reranker", {}).get("service_url")),
        "llm": VLLMClient(base_url=models.get("llm", {}).get("base_url"),
                          model=models.get("llm", {}).get("model")),
        "intent": MiniLMIntentService(models.get("intent", {}).get("service_url")),
        "embedder": EmbeddingService(models.get("embedder", {}).get("service_url")),
        # 离线侧：入湖接口（/v1/index）依赖这两个后端
        "layout_analyzer": LayoutAnalyzerService(models.get("layout", {}).get("service_url")),
        "index_writer": MilvusVectorIndexWriter(
            milvus_uri=vs.get("uri"), collection=vs.get("collection"),
            os_uri=bm.get("uri"), os_index=bm.get("index"),
        ),
    }
