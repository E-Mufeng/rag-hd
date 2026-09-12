# 火电运维 RAG 系统工程框架（docs 总览）

> 本目录 `docs/` 是 `thermal-power-rag` 仓库中**系统框架**的说明文档集合。
> 配套代码骨架位于：`api/`、`pipeline_offline/`、`pipeline_online/`、`configs/`、`mock_services/`、`examples/`。

## 一、项目定位与重要声明

- **场景**：公司内网环境，面向火电维护班组的小范围试点 RAG 问答系统。
- **本仓库性质**：**系统骨架 + 抽象接口 + mock 模拟 + HTTP 服务 + 配置 + 文档**。重在还原真实企业内网小班组试点 RAG 的完整设计思路（架构、业务流程、各环节逻辑）。
- **严格不包含**：
  - ❌ 任何真实业务数据、真实电厂 PDF 规程、真实工单；全部示例均为人工构造的**模拟样例**。
  - ❌ 任何自动下载大模型权重的代码（如 `from_pretrained("Qwen...")`）。
  - ❌ 任何安装 Docker / Milvus / OpenSearch / vLLM / PaddleOCR 的 shell 脚本。
- 所有外部依赖（LLM、向量库、BM25、OCR、rerank、意图模型）均做了**抽象接口层**：业务上层只依赖接口；每依赖提供①真实实现（对接内网服务）②mock 模拟实现。通过 `configs/*.yaml` 切换。

## 二、部署模式

试点阶段采用 **Docker Compose 单机部署**：

| 组件 | 选型 | 说明 |
|---|---|---|
| 向量库（dense/HNSW） | Milvus standalone | 稠密向量 HNSW 索引 |
| BM25 | OpenSearch 容器 | 索引磁盘持久化，重启不丢 |
| LLM | vLLM（Qwen3-14B） | 内网私有化部署，数据不出域；Qwen3 混合推理，问答默认关思维链 |
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
```

详细流程图见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 四、模块说明

| 目录 | 职责 | 关键设计 |
|---|---|---|
| `api/` | HTTP 服务层 | FastAPI：问答/检索/入湖/反馈四接口；配置分层加载 |
| `pipeline_offline/` | 离线文档入湖 | 路由/版面/表格/清洗/语义切分/两级去重/增量索引，全抽象接口+mock |
| `pipeline_online/` | 在线问答 | C1~C5 五段，双闸门、RRF、Top3聚合+Top1兜底、阈值分级 |
| `configs/` | 配置 | `base.yaml`/`docker.yaml`，URI、阈值、模型服务 URL |
| `mock_services/` | 模拟实现 | 向量库/BM25/rerank/LLM/意图 五个 mock，清晰标注"真实部署替换" |
| `examples/` | 演示脚本 | 离线 + 在线两条 mock 流程，仅打印、不真检索/不真调模型 |

## 五、快速体验（mock 模式，无需任何服务）

```bash
PYTHONPATH=. python examples/run_online_demo.py    # 在线问答四例：high/medium/low/领域外拒答
PYTHONPATH=. python examples/run_offline_demo.py   # 离线入湖六步：路由→...→增量索引

PYTHONPATH=. uvicorn api.main:app --port 8080      # 起 HTTP 服务
curl http://127.0.0.1:8080/health
```

## 六、文档导航

- [ARCHITECTURE.md](./ARCHITECTURE.md) — 完整架构与 mermaid 流程图
- [deployment.md](./deployment.md) — 部署说明、寻址口径与 docker-compose 模板
- [case_study.md](./case_study.md) — 项目踩坑全记录

> 生成模型为内网 vLLM 部署的开源 Qwen3-14B，
> 向量化/精排/意图模型均为内网服务化或开源模型（BGE-M3 / BGE-Reranker / MiniLM），
> 通过抽象接口 + 配置切换，详见 `pipeline_online/backends_real.py` 与 `mock_services/`。
