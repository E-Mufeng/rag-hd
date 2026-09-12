# 火电运维 RAG 系统工程框架

> 生成模型为内网 vLLM 部署的开源 **Qwen3-14B**；向量化 / 精排 / 意图均为内网服务化或开源模型（**BGE-M3 / BGE-Reranker-v2-m3 / MiniLM**）。所有外部依赖通过抽象接口 + 配置切换，详见 `pipeline_online/backends_real.py` 与 `mock_services/`。

## 一、项目定位与重要声明

- **场景**：公司内网环境，面向火电维护班组的小范围试点 RAG 问答系统。
- **本仓库性质**：**系统骨架 + 抽象接口 + mock 模拟 + HTTP 服务 + 配置 + 文档**。重在还原真实企业内网小班组试点 RAG 的完整设计思路（架构、业务流程、各环节逻辑）。
- **严格不包含**：
  - ❌ 任何真实业务数据、真实电厂 PDF 规程、真实工单；全部示例均为人工构造的**模拟样例**。
  - ❌ 任何自动下载大模型权重的代码（如 `from_pretrained("Qwen...")`）。
  - ❌ 任何安装 Docker / Milvus / OpenSearch / vLLM / PaddleOCR 的 shell 脚本。
- 所有外部依赖（LLM、向量库、BM25、OCR、rerank、意图模型）均做了**抽象接口层**：业务上层只依赖接口；每依赖提供①真实实现（对接内网服务）②mock 模拟实现。通过 `configs/*.yaml` 切换。

## 二、部署形态

试点阶段采用 **Docker Compose 单机部署**：

| 组件 | 选型 | 说明 |
|---|---|---|
| 向量库（dense/HNSW） | Milvus standalone | 稠密向量 HNSW 索引 |
| BM25 | OpenSearch 容器 | 索引磁盘持久化，重启不丢 |
| LLM | vLLM（Qwen3-14B） | 内网私有化部署，数据不出域；Qwen3 为混合推理模型，运维问答默认关思维链以降首字延迟 |
| 业务服务 | FastAPI（`api/`） | 常驻 HTTP 服务，供班组终端 / 内部系统调用 |

> 后续若试点验证良好、知识库规模上涨，只需改配置即可把 Milvus 升分布式集群、OpenSearch 做集群扩容，**上层业务代码完全不动**。不上 K8s，控制硬件资源消耗。

## 三、架构总览

```
用户提问 ──▶ C1改写(术语规范化/HyDE开关) ──▶ C2意图(MiniLM+规则+_DOMAIN_HINTS+双闸门)
                                                       │ 通过
                                                       ▼
                              C3检索(dense HNSW + BM25 两路 → RRF融合)
                                                       │
                                                       ▼
                              C4精排(CrossEncoder + Top3加权聚合 + Top1兜底 + 阈值分级)
                                                       │ 置信度闸门
                          ┌────────┼────────┐
                        ≥0.7直接答 0.4~0.7中等 <0.4拒答
                                                       ▼
                                    C5生成(Prompt模板 + LLM抽象客户端)

离线：文档 ─▶ 路由 ─▶ 版面分析 ─▶ 表格独立成块 ─▶ 清洗脱敏 ─▶ 语义切分
                            ─▶ 两级去重(MinHash+余弦,分档) ─▶ 增量索引(哈希指纹)

HTTP：api/main.py 把上述链路包装成 /v1/ask、/v1/search、/v1/index、/v1/feedback
```

详细流程图见 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)。

## 四、模块说明

| 目录 | 职责 | 关键设计 |
|---|---|---|
| `api/` | HTTP 服务层 | FastAPI：问答/检索/入湖/反馈四接口；配置分层加载（base + 部署覆盖 + 环境变量） |
| `pipeline_offline/` | 离线文档入湖 | 路由/版面/表格/清洗/语义切分/两级去重/增量索引，全抽象接口+mock |
| `pipeline_online/` | 在线问答 | C1~C5 五段，双闸门、RRF、Top3聚合+Top1兜底、阈值分级；`backends_real.py` 为真实后端对接 |
| `configs/` | 配置 | `base.yaml`（共享默认）/`docker.yaml`（容器内寻址），URI、阈值、模型服务 URL |
| `mock_services/` | 模拟实现 | 向量库/BM25/rerank/LLM/意图 五个 mock，清晰标注"真实部署替换" |
| `examples/` | 演示脚本 | 离线 + 在线 + factory 三条 mock 流程，仅打印、不真检索/不真调模型 |

## 五、快速体验

**方式一：mock 演示（无需任何服务）**

```bash
PYTHONPATH=. python examples/run_online_demo.py     # 在线问答四例：high/medium/low/领域外拒答
PYTHONPATH=. python examples/run_offline_demo.py    # 离线入湖六步：路由→...→增量索引
PYTHONPATH=. python examples/run_with_factory.py    # 用 factory 装配后端、端到端跑通
```

**方式二：起 HTTP 服务（同样默认 mock，无需内网服务）**

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8080
curl http://127.0.0.1:8080/health
```

真实部署把 `configs/base.yaml` 的 `run.use_mock` 置为 `false`（或用环境变量 `RAG_USE_MOCK=false`），
并把地址换成内网实际地址，业务代码零改动。

## 六、接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活探针 + 后端装配状态（compose healthcheck 用） |
| GET | `/v1/meta` | 当前生效配置摘要（现场核对连的哪个库、哪个模型） |
| POST | `/v1/ask` | 在线问答主入口，返回答案 + 置信度分级 + 来源片段 |
| POST | `/v1/search` | 仅检索（不调生成模型），用于排查召回问题 |
| POST | `/v1/index` | 离线入湖，`version` 变更触发删旧 chunk |
| POST | `/v1/feedback` | 维护班反馈回流，供术语库与提示词迭代 |

> 拒答不是错误：意图闸门 / 低置信拒答均返回 HTTP 200 + `rejected=true` 与 `reject_reason`，
> 便于调用方区分「系统按策略拒绝作答」与「服务故障」。

## 七、mock 与真实后端的区分

| 抽象接口（业务只依赖它） | mock 实现（`mock_services/`） | 真实实现（`pipeline_online/backends_real.py`） |
|---|---|---|
| `VectorStore` | `MockVectorStore`（哈希向量） | `MilvusVectorStore`(HNSW) |
| `BM25Store` | `MockBM25`（简化 BM25） | `OpenSearchBM25` |
| `Reranker` | `MockReranker`（词重叠） | `BGERerankerService`（BGE-Reranker） |
| `LLMClient` | `MockLLM`（模板回显） | `VLLMClient`（Qwen3-14B） |
| `IntentModel` | `MockIntentModel`（规则） | `MiniLMIntentService`（MiniLM 服务） |
| `Embedder` | `MockEmbedder`（哈希） | `EmbeddingService`（BGE-M3 服务） |
| `LayoutAnalyzer` | `MockLayoutAnalyzer` | `LayoutAnalyzerService`（内网版面/OCR） |
| `VectorIndexWriter` | `MockIncrementalIndexer` | `MilvusVectorIndexWriter`（Milvus + OpenSearch 双写） |

切换由 `pipeline_online/factory.build_backends(cfg)` 完成：读 `configs/*.yaml` 的 `run.use_mock`，返回对应实例，**业务代码零改动**。

## 八、文档导航

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — 完整架构与 mermaid 流程图
- [docs/deployment.md](./docs/deployment.md) — 部署说明、寻址口径与 docker-compose 模板
- [docs/case_study.md](./docs/case_study.md) — 项目踩坑全记录
