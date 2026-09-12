"""
schemas.py —— HTTP 接口的请求 / 响应模型（Pydantic）

为什么单独定义：接口契约与内部管线解耦。管线内部用 dataclass 传递数据，
对外只暴露稳定的 JSON 结构，前端 / 企业微信 / 内部系统的调用方不必了解内部字段。
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 健康检查 / 元信息
# ============================================================================
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mode: str = Field(description="部署形态，固定 docker")
    use_mock: bool = Field(description="true=全部 mock 后端；false=真实内网服务")
    backends: dict = Field(default_factory=dict, description="各后端实现类名")


class MetaResponse(BaseModel):
    project: str
    mode: str
    use_mock: bool
    llm_model: str
    vectorstore: str
    bm25: str
    confidence: dict
    retrieval: dict


# ============================================================================
# 在线问答
# ============================================================================
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户原始问题")
    top_k: int = Field(5, ge=1, le=20, description="精排后送入生成的片段数")
    thinking: bool = Field(False, description="是否开启 Qwen3 思维链；运维问答默认关闭以降低首字延迟")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "汽轮机润滑油压力低报警值是多少？", "top_k": 5, "thinking": False}]
        }
    }


class SourceItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None


class IntentInfo(BaseModel):
    label: str
    confidence: float
    passed_pre_gate: bool


class AskResponse(BaseModel):
    request_id: str
    answer: str
    level: Literal["high", "medium", "low"] = Field(description="置信度分级")
    confidence: float = Field(description="检索置信度（意图闸门拦截时为 0；意图自身置信度见 intent.confidence）")
    rejected: bool = Field(description="是否拒答")
    reject_reason: Optional[str] = Field(None, description="intent_gate / low_confidence / None")
    intent: Optional[IntentInfo] = None
    sources: List[SourceItem] = Field(default_factory=list)
    latency_ms: int


# ============================================================================
# 仅检索（供运营 / 调试观察召回效果，不调用生成模型）
# ============================================================================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)


class SearchResponse(BaseModel):
    request_id: str
    query_rewritten: str
    items: List[SourceItem]
    rrf_k: int
    latency_ms: int


# ============================================================================
# 离线入湖（文档入库）
# ============================================================================
class IndexRequest(BaseModel):
    paths: List[str] = Field(..., min_length=1, description="待入湖文档路径（PDF / Word / 扫描件）")
    version: str = Field(..., description="文档版本号，参与增量索引指纹，版本变更即触发删旧")
    max_chars: int = Field(800, ge=200, le=4000, description="语义切分窗口上限")


class IndexResponse(BaseModel):
    request_id: str
    routed: int = Field(description="已路由文档数")
    rejected: List[str] = Field(default_factory=list, description="不支持的类型（拒收）")
    chunks: int = Field(description="切分出的 chunk 总数")
    written: List[str] = Field(default_factory=list, description="实际新增/更新的 chunk_id")
    dedup: dict = Field(default_factory=dict, description="去重分档统计")
    latency_ms: int


# ============================================================================
# 维护班反馈（答案纠错回流）
# ============================================================================
class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="对应 AskResponse.request_id")
    helpful: bool
    corrected_answer: str = Field("", description="维护班给出的正确答复（可选）")
    comment: str = Field("", description="补充说明（可选）")


class FeedbackResponse(BaseModel):
    accepted: bool
    stored: bool = True
