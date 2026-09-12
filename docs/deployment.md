# 部署说明（thermal-power-rag 工程框架）

> 本文档描述试点部署的组件构成、寻址口径、配置注入与 docker-compose 模板。
> **本项目不提供任何镜像、不提供一键启动脚本、不写安装命令**，仅给出模板作为部署参考。

## 1. 部署形态

试点阶段采用 **Docker Compose 单机部署**，组件如下：

| 组件 | 选型 | 说明 |
|---|---|---|
| 向量库 | Milvus standalone | 稠密向量 HNSW 索引 |
| BM25 | OpenSearch 容器 | 索引磁盘持久化，服务重启不丢索引 |
| LLM | vLLM（Qwen3-14B） | 内网私有化部署，数据不出域；Qwen3 为混合推理模型，问答默认关思维链降首字延迟 |
| 业务服务 | FastAPI（`api/`） | 常驻 HTTP 服务：问答 / 检索 / 入湖 / 反馈 |
| 模型服务 | embed / rerank / layout | 向量化、精排、版面 OCR，通常部署在 GPU 推理节点 |

不上 K8s 分布式集群，控制硬件资源消耗，适配小班组试点规模。
后续若知识库规模上涨，只需修改配置即可把 Milvus 升分布式集群、OpenSearch 做集群扩容，**上层业务代码完全不动**。

## 2. 前提条件

- 服务器资源充足：Milvus standalone 建议 ≥4C8G；OpenSearch ≥2C4G；vLLM 跑 14B 需 GPU（FP16 约 28GB / 4bit 约 10GB）。
- **内网私有 Harbor 已预置镜像**：`milvusdb/milvus`、`opensearchproject/opensearch`、`vllm/vllm-openai`、以及业务镜像 `thermal-power-rag-api`。本项目不提供镜像，也不编写拉取/构建镜像的脚本。
- 业务镜像构建命令示例（构建由 CI 执行，本仓库不提供脚本）：
  `uvicorn api.main:app --host 0.0.0.0 --port 8080`

## 3. 寻址口径（重要，最容易返工的地方）

同一服务在不同视角下地址完全不同：

| 视角 | 写法 | 示例 | 说明 |
|---|---|---|---|
| 容器内互访 | compose 服务名 | `http://milvus:19530` | 由 Docker 内建 DNS 解析，**只在同一 network 内的容器里可用** |
| 宿主机 / 跨机 | 宿主 IP 或内网域名 | `http://10.10.20.11:19530` | 走 compose 的 `ports` 映射出来的端口 |
| 本机自测 | 回环地址 | `http://127.0.0.1:19530` | 仅宿主机上进程可用 |

**所以：**
- `configs/docker.yaml` 里写的是容器内视角（`http://milvus:19530`），给 `rag-api` 容器用，正确；
  但在宿主机浏览器里敲 `http://milvus:19530` 一定打不开——这不是配置错误，是视角不同。
- `configs/base.yaml` 里写的是内网地址占位（`http://10.10.20.11:19530`），给宿主机直连调试用。
- 真实部署前，把地址按现场情况改成实际值；**推荐用环境变量注入而不是改文件**（见下一节）。

## 4. 配置分层与注入

配置优先级（后者覆盖前者）：

1. `configs/base.yaml` —— 共享默认（阈值、检索参数、地址占位）
2. `configs/{deployment.mode}.yaml` —— 部署形态覆盖（当前固定 `docker.yaml`）
3. `RAG_*` 环境变量 —— 编排层注入（最高优先级）

支持的环境变量：

| 变量 | 覆盖的配置项 |
|---|---|
| `RAG_CONFIG_DIR` | 配置目录（容器内可挂载覆盖） |
| `RAG_USE_MOCK` | `run.use_mock`（true / false） |
| `RAG_MILVUS_URI` | `storage.vectorstore.uri` |
| `RAG_OPENSEARCH_URI` | `storage.bm25.uri` |
| `RAG_LLM_BASE_URL` | `models.llm.base_url` |
| `RAG_LLM_MODEL` | `models.llm.model` |
| `RAG_EMBEDDER_URL` / `RAG_INTENT_URL` / `RAG_RERANKER_URL` / `RAG_LAYOUT_URL` | 各模型服务地址 |
| `RAG_LOG_DIR` | 拒答 / 低置信 / 反馈日志目录 |

这样做的意义：**同一份镜像可以在试点机、扩容后的集群、测试环境之间漂移，只改编排层的环境变量，不重打镜像。**

## 5. docker-compose 模板

> 仅展示服务定义与依赖关系，作为部署参考。**镜像需在 Harbor 预置；本项目不提供镜像、不提供启动脚本。**

```yaml

services:
  milvus:
    image: milvusdb/milvus:latest          # 内网 Harbor 预置等价镜像
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus        # 向量持久化
    environment:
      - ETCD_USE_EMBED=true
      - MINIO_USE_EMBED=true

  opensearch:
    image: opensearchproject/opensearch:latest   # 内网 Harbor 预置等价镜像
    ports:
      - "9200:9200"
    volumes:
      - os_data:/usr/share/opensearch/data       # BM25 索引磁盘持久化
    environment:
      - discovery.type=single-node
      - "OPENSEARCH_INITIAL_ADMIN_PASSWORD=CHANGE_ME_STRONG_PWD"  # 试点内部约定，生产按内网规范

  vllm:                                        # 可选：也可部署在独立 GPU 节点
    image: vllm/vllm-openai:latest             # 内网 Harbor 预置等价镜像
    command: >
      --model Qwen/Qwen3-14B
      --served-model-name Qwen3-14B
      --dtype auto --max-model-len 8192
      --gpu-memory-utilization 0.9
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  rag-api:                                     # 业务服务
    image: harbor.internal/rag/thermal-power-rag-api:0.1.0   # 业务镜像由 CI 构建后推 Harbor
    command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"
    environment:
      - RAG_USE_MOCK=false
      - RAG_MILVUS_URI=http://milvus:19530
      - RAG_OPENSEARCH_URI=http://opensearch:9200
      - RAG_LLM_BASE_URL=http://vllm:8000/v1
      - RAG_LLM_MODEL=Qwen3-14B
      - RAG_EMBEDDER_URL=http://embed:8001/v1/embeddings
      - RAG_INTENT_URL=http://embed:8002/v1/embeddings
      - RAG_RERANKER_URL=http://rerank:8003/v1/rank
      - RAG_LAYOUT_URL=http://layout:8004/v1/layout
      - RAG_LOG_DIR=/data/logs
    volumes:
      - ./data/logs:/data/logs
    depends_on: [milvus, opensearch]

volumes:
  milvus_data:
  os_data:
```

> 注意 `--served-model-name` 必须与 `configs` 里的 `llm.model` 一致，否则调用时会 404。
> Qwen3 若需服务端解析思维链输出，追加 `--reasoning-parser qwen3`；本项目默认关闭思维链。

## 6. 健康检查与联调

```bash
# 容器内 / 宿主机（按视角替换地址）
curl http://127.0.0.1:8080/health        # status=ok 且 backends 列出各实现类名
curl http://127.0.0.1:8080/v1/meta       # 核对当前连的库、模型、阈值

curl -X POST http://127.0.0.1:8080/v1/ask \
     -H "Content-Type: application/json" \
     -d '{"query":"汽轮机润滑油压力低报警值是多少？","top_k":5}'
```

`/health` 返回 `degraded` 说明有后端未装配成功；真实模式下它**不会主动连中间件**（避免探针拖慢或误判），
连接问题会在实际调用时以异常形式暴露。

## 7. 水平演进路径

- 知识库规模上涨 → 修改配置把 Milvus 升为分布式集群、OpenSearch 做集群扩容，**业务代码不动**。
- 生成精度不足 → 调 vLLM 侧参数（量化策略、tensor-parallel-size），或升级 GPU。
- 检索漏召 → 调大 `retrieval.dense_top_k` / `bm25_top_k`；粗排保持宽召回（`recall_priority: true`）。
- 阈值不合适 → 改 `configs/base.yaml` 的 `confidence` 段，配合现场抽样复核后定稿，无需改代码。

## 8. 重要声明

- 本框架**不真实拉起任何服务**；仓库内示例默认走 mock 模式。
- 不写任何 Dockerfile 构建、镜像拉取、模型下载脚本。
- 真实部署在内网服务就绪后进行；本地无服务时调用真实后端会抛连接错误，属预期。
