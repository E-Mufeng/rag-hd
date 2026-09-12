"""
main.py —— FastAPI 服务入口（在线问答 / 检索 / 入湖 / 反馈）

为什么要有这一层：pipeline_* 是库（能力），examples/ 是脚本（演示），
真正让班组成员用起来的是「常驻 HTTP 服务」——班组终端、企业微信机器人、
检修管理系统都通过 HTTP 调用它。没有这一层，框架只能自己跑着玩。

服务启动（工作目录为仓库根，使 pipeline_* 可被导入）：
    PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8080

接口一览：
    GET  /health        存活探针 + 后端装配状态（供 compose healthcheck 使用）
    GET  /v1/meta       当前生效配置摘要（运维核对用：模式、模型名、阈值）
    POST /v1/ask        在线问答主入口（C1→C5 全链路）
    POST /v1/search     仅检索（调参与召回排查用，不调用生成模型）
    POST /v1/index      离线入湖（A1→A9，文档入库 / 版本更新）
    POST /v1/feedback   维护班反馈回流（落盘，供术语库与领域提示词迭代）

设计取舍：
    - 端点用同步 def：内部是阻塞式 HTTP 调用（内网服务），FastAPI 会自动丢到
      线程池执行，不阻塞事件循环；若改成 async def 反而会卡死整个进程。
    - 拒答不是错误：意图闸门 / 低置信拒答都返回 200 + 结构化字段
      （rejected / reject_reason），让调用方能分辨「系统拒绝作答」与「服务故障」。
    - 不在此层做重试与降级：重试属客户端或网关职责，服务层保持语义单一。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import schemas
from api.config import load_config
from api.service import RagService
from pipeline_online import factory

logger = logging.getLogger("rag.api")

_REQUIRED_BACKENDS = ("vectorstore", "bm25", "reranker", "llm", "intent", "embedder",
                     "layout_analyzer", "index_writer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时装载配置与后端；整套后端只在进程启动时装配一次，请求内复用。"""
    cfg = load_config()
    backends = factory.build_backends(cfg)
    app.state.cfg = cfg
    app.state.backends = backends
    app.state.service = RagService(cfg, backends)
    logger.info("RAG 服务已装配：mode=%s use_mock=%s", cfg.get("deployment", {}).get("mode"),
                cfg.get("run", {}).get("use_mock"))
    yield
    # 进程退出无需显式释放：真实后端均为短连接、方法内按需创建客户端


app = FastAPI(
    title="火电运维 RAG 服务",
    version="0.1.0",
    description="内网私有化部署的火电运维知识库问答服务（维护班组试点）",
    lifespan=lifespan,
)

# 内网试点：调用方为班组终端与内部系统，先放开跨域；对外提供服务时按内网规范收紧 allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """统一异常出口：对外只给可读信息，堆栈留在服务端日志。"""
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误，请联系运维查看服务端日志"},
    )


def _service(request: Request) -> RagService:
    return request.app.state.service


# ============================================================================
# 探针与元信息
# ============================================================================
@app.get("/health", response_model=schemas.HealthResponse, tags=["ops"])
def health(request: Request):
    """存活探针。真实模式下不主动连中间件（避免探针拖慢/误判），只报装配状态。"""
    cfg = request.app.state.cfg
    backends = request.app.state.backends
    missing = [k for k in _REQUIRED_BACKENDS if not backends.get(k)]
    return schemas.HealthResponse(
        status="ok" if not missing else "degraded",
        mode=cfg.get("deployment", {}).get("mode", "docker"),
        use_mock=bool(cfg.get("run", {}).get("use_mock", True)),
        backends={k: type(v).__name__ for k, v in backends.items() if v is not None},
    )


@app.get("/v1/meta", response_model=schemas.MetaResponse, tags=["ops"])
def meta(request: Request):
    """返回当前生效配置摘要，便于现场核对「这份服务连的是哪个库、哪个模型」。"""
    cfg = request.app.state.cfg
    return schemas.MetaResponse(
        project=cfg.get("project", {}).get("name", "thermal-power-rag"),
        mode=cfg.get("deployment", {}).get("mode", "docker"),
        use_mock=bool(cfg.get("run", {}).get("use_mock", True)),
        llm_model=cfg.get("models", {}).get("llm", {}).get("model", ""),
        vectorstore=cfg.get("storage", {}).get("vectorstore", {}).get("backend", ""),
        bm25=cfg.get("storage", {}).get("bm25", {}).get("backend", ""),
        confidence=cfg.get("confidence", {}),
        retrieval=cfg.get("retrieval", {}),
    )


# ============================================================================
# 业务接口
# ============================================================================
@app.post("/v1/ask", response_model=schemas.AskResponse, tags=["rag"])
def ask(req: schemas.AskRequest, request: Request):
    """
    在线问答。返回 answer + 置信度分级 + 来源片段。

    调用方需按 level 分别处理：
        high   —— 可直接展示；
        medium —— 展示时保留「请核实」提示，不要摘掉；
        low    —— rejected=true，answer 为拒答话术，属正常业务返回而非故障。
    """
    return _service(request).ask(req)


@app.post("/v1/search", response_model=schemas.SearchResponse, tags=["rag"])
def search(req: schemas.SearchRequest, request: Request):
    """仅检索（两路召回 + RRF 融合），不调用生成模型；用于排查「该被召回却没召回」。"""
    return _service(request).search(req)


@app.post("/v1/index", response_model=schemas.IndexResponse, tags=["rag"])
def index(req: schemas.IndexRequest, request: Request):
    """离线入湖：文档路由→版面→清洗→切分→去重→增量索引。version 变更会触发删旧 chunk。"""
    return _service(request).index(req)


@app.post("/v1/feedback", response_model=schemas.FeedbackResponse, tags=["rag"])
def feedback(req: schemas.FeedbackRequest, request: Request):
    """维护班反馈回流：记录答复是否可用、正确答复与备注，供后续术语库/提示词迭代。"""
    return _service(request).feedback(req)
