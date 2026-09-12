# 完整目录树与说明（thermal-power-rag 工程框架）

> 本文件列出本框架的全部目录/文件，并标注「哪些是 mock 占位 / 真实落地如何对接内网服务」。

## 1. 目录树

```
thermal-power-rag/
├── docker-compose.yaml            # 顶层编排模板（模板，镜像需 Harbor 预置，不提供启动脚本）
├── requirements.txt               # 依赖清单（仅列依赖名）
├── README.md                      # 项目入口
├── PROJECT_FRAMEWORK_TREE.md      # 本文件
│
├── api/                           # HTTP 服务层（FastAPI，常驻服务）
│   ├── __init__.py
│   ├── config.py                  # 配置分层加载：base + {mode}.yaml + RAG_* 环境变量
│   ├── schemas.py                 # 请求/响应模型（对外契约）
│   ├── service.py                 # 用例编排：ask / search / index / feedback
│   └── main.py                    # FastAPI 应用与路由（含 lifespan 装配、统一异常出口）
│
├── pipeline_offline/              # 离线文档入湖管线（骨架+抽象+mock）
│   ├── __init__.py
│   ├── router.py                  # A1 文档分发路由（PDF/Word/扫描件OCR路由）
│   ├── layout_analyzer.py         # A2 版面分析【抽象接口】+ MockLayoutAnalyzer
│   ├── table_parser.py            # A3 表格独立成块+元数据【抽象+mock】
│   ├── cleaner.py                 # A6 清洗/脱敏（纯函数，可直接用）
│   ├── semantic_chunker.py        # A7 语义切分（标题优先/窗口兜底，chunk_id 全局唯一）
│   ├── embedder.py                # 向量化抽象 Embedder：MockEmbedder + EmbeddingService(BGE-M3 内网服务)
│   ├── dedup.py                   # A8 两级去重（MinHash+余弦，阈值分档 auto_merge/need_review/keep_both）
│   └── incremental_index.py       # A9 增量索引（content+version 指纹，删旧 chunk）
│
├── pipeline_online/               # 在线问答管线（C1~C5）
│   ├── __init__.py
│   ├── rewrite.py                 # C1 问题改写（术语规范化 + HyDE 开关，默认关）
│   ├── intent.py                  # C2 意图（MiniLM抽象+规则兜底+_DOMAIN_HINTS+双闸门+拒答日志）
│   ├── retrieve.py                # C3 两路检索（dense HNSW + BM25）+ RRF 融合【抽象 VectorStore/BM25Store】
│   ├── rerank.py                  # C4 精排（CrossEncoder抽象+Top3加权聚合+Top1兜底+阈值分级 0.7/0.4）
│   ├── generate.py                # C5 生成（Prompt模板+LLM抽象客户端 vLLM）
│   ├── backends_real.py           # 【真实后端】Milvus / OpenSearch / vLLM / BGE-Reranker / MiniLM / 版面服务 / 索引双写
│   └── factory.py                 # build_backends(cfg)：读 configs 装配 mock/real 实例，业务零改动切换
│
├── configs/                       # 配置
│   ├── base.yaml                  # 共享默认（阈值/模型URL/部署参数/日志）
│   └── docker.yaml                # 部署覆盖：容器内寻址（Milvus(HNSW)+OpenSearch+vLLM）
│
├── mock_services/                 # 全部外部依赖的模拟实现（仅演示，真实部署替换）
│   ├── __init__.py
│   ├── mock_vectorstore.py        # 模拟稠密向量检索（HNSW 抽象）
│   ├── mock_bm25.py               # 模拟 BM25 检索（OpenSearch 抽象）
│   ├── mock_rerank.py             # 模拟 CrossEncoder 精排打分
│   ├── mock_llm.py                # 模拟 LLM 生成（vLLM 抽象）
│   └── mock_intent.py             # 模拟 MiniLM 意图分类
│
├── docs/                          # 文档集合
│   ├── README.md                  # 框架总览（定位/部署模式/快速体验）
│   ├── ARCHITECTURE.md            # 架构说明 + mermaid 流程图 + 接口对照
│   ├── deployment.md              # 部署说明 + 寻址口径 + docker-compose 模板
│   └── case_study.md              # 踩坑记录（OCR/BGE-M3/意图边界/Rerank假阳性/阈值）
│
├── data/                          # 运行时目录（不入库，见 .gitignore）
│   └── logs/                      # 拒答 / 低置信 / 反馈日志（维护班回流用）
│
└── examples/                      # 演示脚本（mock 模式，仅打印流程）
    ├── run_offline_demo.py        # 离线六步：路由→...→增量索引
    ├── run_online_demo.py         # 在线四例：high/medium/low/领域外前置拒答
    └── run_with_factory.py        # 用 factory 装配后端端到端跑通（演示 mock/real 切换）
```

## 2. 各段说明：mock 占位 vs 真实落地对接

| 模块 | mock / 占位（本框架） | 真实落地（对接内网服务） |
|---|---|---|
| `pipeline_offline/router.py` | 按扩展名初判；扫描件置 need_ocr | 扫描件真实走 OCR 适配器 |
| `layout_analyzer.LayoutAnalyzer` | `MockLayoutAnalyzer` 返回结构化骨架 | `LayoutAnalyzerService` → 内网 PP-Structure / 商业 OCR 版面还原 |
| `table_parser.TableParser` | `MockTableParser` 转可读文本+元数据 | 复杂表结构还原（合并单元格/跨页） |
| `cleaner.py` | 规则正则脱敏（工号/手机/页码） | 按内网数据合规扩充字典 |
| `semantic_chunker.py` | 标题正则+窗口兜底，纯逻辑 | 可接 embedding 做语义边界（可选） |
| `embedder.Embedder` | `MockEmbedder` 确定性哈希向量 | `EmbeddingService` → 内网 BGE-M3 服务（dense 1024） |
| `dedup.TwoStageDedup` | `MockEmbedder` 哈希向量 | 注入 `EmbeddingService`（内网 BGE-M3 余弦） |
| `incremental_index.VectorIndexWriter` | `MockIncrementalIndexer` 内存指纹表 | `MilvusVectorIndexWriter` → Milvus upsert(HNSW) + OpenSearch 索引（删旧指纹） |
| `rewrite.HydeGenerator` | 接口占位，默认关闭 | 内网 LLM 生成假设文档 |
| `intent.IntentModel` | `MockIntentModel` 规则 / `MiniLMIntentService` 真实 | `MiniLMIntentService` → 内网 MiniLM embedding 服务 + 模板最近邻 |
| `intent._DOMAIN_HINTS` | 内置强领域/领域外词表 | 维护班回流持续扩充 |
| `retrieve.VectorStore` | `MockVectorStore` | `MilvusVectorStore`（HNSW） |
| `retrieve.BM25Store` | `MockBM25` | `OpenSearchBM25` |
| `rerank.Reranker` | `MockReranker` | `BGERerankerService` → BGE-Reranker-v2-m3（内网 rerank 服务） |
| `generate.LLMClient` | `MockLLM` | `VLLMClient` → vLLM（Qwen3-14B）OpenAI 兼容 |
| `api/*` | 全部走 mock 后端，可直接起服务 | 同代码，靠 `run.use_mock=false` + 环境变量切内网地址 |

## 3. 运行验证（mock，无需任何服务）

```bash
# 脚本方式
PYTHONPATH=. python examples/run_online_demo.py     # 在线四例：覆盖 high/medium/low/领域外前置拒答
PYTHONPATH=. python examples/run_offline_demo.py    # 离线六步：路由→...→增量索引（含去重分档）
PYTHONPATH=. python examples/run_with_factory.py    # 用 factory 装配后端端到端跑通

# 服务方式
PYTHONPATH=. uvicorn api.main:app --port 8080
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/ask \
     -H "Content-Type: application/json" \
     -d '{"query":"汽轮机润滑油压力低报警值是多少？"}'
```

## 4. 配置切换

- `run.use_mock: true` → 全部后端走 mock，不连任何服务，用于静态阅读 / 流程演示 / 接口联调。
- `run.use_mock: false` → 真实后端：Milvus + OpenSearch + vLLM + BGE-Reranker + MiniLM + BGE-M3 + 版面服务。
- 配置分层：`configs/base.yaml`（共享默认）→ `configs/{deployment.mode}.yaml`（部署覆盖）→ `RAG_*` 环境变量（最高优先级）。
- 地址口径：容器内互访用服务名（`http://milvus:19530`），宿主机 / 跨机访问用内网 IP（见 `docs/deployment.md`）。

## 5. 已知点 / 待确认

- 生成模型为内网 vLLM 部署的开源 Qwen3-14B，向量化 / 精排 / 意图均为内网服务化或开源模型。
- 真实后端类（`backends_real.py`）通过 HTTP / 内网服务 / 中间件客户端对接，**不下载任何模型权重**；重依赖（pymilvus / opensearch-py / openai）均方法内懒加载，模块可安全 import，mock 演示不依赖它们。
- 所有 mock 评分为确定性近似，仅用于走通流程；真实阈值（`confidence` / `dedup`）需结合维护班现场复核重新标定。
- `data/logs/` 下日志为运行时产物，不随仓库提交；线上靠这些日志回流优化领域提示词与术语表。
