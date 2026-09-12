"""
service.py —— 用例编排层（API 与管线之间的唯一粘合层）

职责划分：
    api/main.py   只做「HTTP 协议」的事：解析请求、校验、状态码、序列化；
    api/service.py 只做「用例编排」的事：把 C1~C5 串成一次问答，把离线模块串成一次入湖；
    pipeline_*     只做「单一能力」的事：检索、精排、生成、切分、去重……

这样分的好处：同一套用例逻辑既可以被 HTTP 调用，也可以被定时任务 / CLI / 消息
中间件消费者直接调用，不需要复制一遍流程代码。

编排逻辑与 examples/run_with_factory.py 保持一致（同一链路，两种调用入口），
差异仅在：本文件额外处理分级话术、来源溯源、耗时统计与埋点落盘。
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from api import schemas
from pipeline_offline.cleaner import clean_blocks
from pipeline_offline.dedup import DedupTier, TwoStageDedup
from pipeline_offline.layout_analyzer import LayoutBlock
from pipeline_offline.router import DocType, route
from pipeline_offline.semantic_chunker import chunk_block
from pipeline_online.generate import GenerationInput, generate
from pipeline_online.intent import detect_intent
from pipeline_online.rerank import rerank
from pipeline_online.retrieve import retrieve
from pipeline_online.rewrite import rewrite

# 拒答话术（统一出口，避免各处措辞不一致）
REJECT_ANSWER = (
    "抱歉，当前资料不足以给出可靠答复。为避免误导操作，此处不作答。\n"
    "建议：补充设备编号或规程名称后重试，或直接联系运维班组核实。"
)
# 中等置信前缀
MEDIUM_PREFIX = "【提示】以下答复基于现有资料给出，资料覆盖度有限，请与运行规程原文核对后使用。\n\n"


class RagService:
    """RAG 用例编排：在线问答 / 仅检索 / 离线入湖 / 反馈回流。"""

    def __init__(self, cfg: Dict[str, Any], backends: Dict[str, Any]):
        self.cfg = cfg or {}
        self.backends = backends or {}
        self.log_dir = self._resolve_log_dir()

    # ------------------------------------------------------------------ 内部
    def _resolve_log_dir(self) -> str:
        log_cfg = self.cfg.get("logging", {}) or {}
        if log_cfg.get("dir"):
            return log_cfg["dir"]
        # 兼容 base.yaml 中以具体文件路径声明日志的写法
        reject_log = log_cfg.get("reject_log")
        if reject_log:
            return os.path.dirname(reject_log) or "./data/logs"
        return "./data/logs"

    @property
    def _confidence(self) -> Dict[str, float]:
        return self.cfg.get("confidence", {}) or {}

    @property
    def _retrieval(self) -> Dict[str, Any]:
        return self.cfg.get("retrieval", {}) or {}

    def _write_log(self, name: str, payload: Dict[str, Any]) -> None:
        """运行时埋点落盘（拒答 / 反馈）。失败不影响主流程。"""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, name), "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _to_sources(self, decision) -> List[schemas.SourceItem]:
        out: List[schemas.SourceItem] = []
        for r in decision.top_results:
            meta = getattr(r, "meta", None) or {}
            out.append(schemas.SourceItem(
                chunk_id=r.chunk_id,
                text=r.text,
                score=round(float(r.score), 4),
                page=meta.get("page"),
                section=meta.get("section"),
            ))
        return out

    # ------------------------------------------------------------ 在线问答
    def ask(self, req: schemas.AskRequest) -> schemas.AskResponse:
        """C1 改写 → C2 意图 → C3 检索 → C4 精排 → C5 生成。"""
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        threshold = (self.cfg.get("intent", {}) or {}).get("threshold", 0.55)

        # C1 问题改写
        rw = rewrite(req.query, use_hyde=bool((self.cfg.get("rewrite", {}) or {}).get("hyde_enabled", False)))

        # C2 意图识别（前置闸门）
        intent = detect_intent(rw.rewritten, model=self.backends.get("intent"),
                               threshold=threshold, log_dir=self.log_dir)
        intent_info = schemas.IntentInfo(label=intent.label, confidence=intent.confidence,
                                         passed_pre_gate=intent.passed_pre_gate)
        if not intent.passed_pre_gate:
            return self._reject(request_id, "intent_gate", intent_info, t0)

        # C3 两路检索 + RRF 融合
        cands = retrieve(rw.rewritten, self.backends["vectorstore"], self.backends["bm25"],
                         dense_top_k=self._retrieval.get("dense_top_k", 10),
                         bm25_top_k=self._retrieval.get("bm25_top_k", 10),
                         rrf_k=self._retrieval.get("rrf_k", 60))
        pairs = [(c.chunk_id, c.text) for c in cands[:req.top_k]]

        # C4 精排 + 置信度闸门
        decision = rerank(rw.rewritten, pairs, self.backends["reranker"],
                          top_n=req.top_k, log_dir=self.log_dir)
        if decision.level == "low":
            return self._reject(request_id, "low_confidence", intent_info, t0)

        # C5 生成
        llm = self.backends["llm"]
        if req.thinking and hasattr(llm, "enable_thinking"):
            # Qwen3 混合推理：按请求临时开启思维链；浅拷贝避免污染共享客户端实例
            llm = copy.copy(llm)
            llm.enable_thinking = True
        answer = generate(GenerationInput(req.query, decision, [t for _, t in pairs]),
                          client=llm)
        if decision.level == "medium":
            answer = MEDIUM_PREFIX + answer

        return schemas.AskResponse(
            request_id=request_id,
            answer=answer,
            level=decision.level,
            confidence=decision.confidence,
            rejected=False,
            intent=intent_info,
            sources=self._to_sources(decision),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    def _reject(self, request_id: str, reason: str,
                intent_info: Optional[schemas.IntentInfo], t0: float) -> schemas.AskResponse:
        return schemas.AskResponse(
            request_id=request_id,
            answer=REJECT_ANSWER,
            level="low",
            # confidence 的语义统一为「检索置信度」：意图闸门拦截时未做检索，故为 0；
            # 意图自身的置信度在 intent 字段里，不要混用这两个值。
            confidence=0.0,
            rejected=True,
            reject_reason=reason,
            intent=intent_info,
            sources=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # -------------------------------------------------------------- 仅检索
    def search(self, req: schemas.SearchRequest) -> schemas.SearchResponse:
        """只跑 C1~C3（不精排、不生成），用于观察召回是否漏掉关键规程片段。"""
        t0 = time.perf_counter()
        rw = rewrite(req.query)
        cands = retrieve(rw.rewritten, self.backends["vectorstore"], self.backends["bm25"],
                         dense_top_k=self._retrieval.get("dense_top_k", 10),
                         bm25_top_k=self._retrieval.get("bm25_top_k", 10),
                         rrf_k=self._retrieval.get("rrf_k", 60))
        items = [schemas.SourceItem(chunk_id=c.chunk_id, text=c.text,
                                    score=round(float(c.rrf_score), 6),
                                    page=(c.meta or {}).get("page"),
                                    section=(c.meta or {}).get("section"))
                 for c in cands[:req.top_k]]
        return schemas.SearchResponse(
            request_id=uuid.uuid4().hex[:12],
            query_rewritten=rw.rewritten,
            items=items,
            rrf_k=self._retrieval.get("rrf_k", 60),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ------------------------------------------------------------ 离线入湖
    def index(self, req: schemas.IndexRequest) -> schemas.IndexResponse:
        """A1 路由 → A2 版面 → A6 清洗 → A7 切分 → A8 去重 → A9 增量索引。"""
        t0 = time.perf_counter()
        analyzer = self.backends.get("layout_analyzer")
        writer = self.backends.get("index_writer")

        routed_cnt = 0
        rejected: List[str] = []
        chunks = []

        for path in req.paths:
            r = route(path)
            if r.doc_type == DocType.UNKNOWN:
                rejected.append(path)
                continue
            routed_cnt += 1
            # doc_key 取文件名（去扩展名）：chunk_id 需带文档标识才能跨文档唯一
            doc_key = os.path.splitext(os.path.basename(path))[0]
            seq = 0   # 文档级序号，跨版面块累加，避免同一文档内 chunk_id 撞车
            if r.need_ocr:
                text = analyzer.ocr(path)
                blocks = [LayoutBlock(block_type="text", content=text, page=1,
                                      meta={"document": os.path.basename(path)})]
            else:
                blocks = analyzer.analyze(path)
                for b in blocks:
                    b.meta.setdefault("document", os.path.basename(path))
            for b in clean_blocks(blocks):
                cs = chunk_block(b, max_chars=req.max_chars, doc_key=doc_key, seq_offset=seq)
                seq += len(cs)
                chunks.extend(cs)

        # 两级去重：自动合并档位的副本不进索引，待复核档位照常入库并挂复核标记
        dedup_cfg = self.cfg.get("dedup", {}) or {}
        dedup = TwoStageDedup(
            minhash_threshold=dedup_cfg.get("minhash_threshold", 0.85),
            embedding_high=dedup_cfg.get("embedding_high", 0.95),
            embedding_medium=dedup_cfg.get("embedding_medium", 0.85),
            embedder=self.backends.get("embedder"),
        )
        decisions = dedup.run(chunks)
        merged_ids = {d.chunk_id for d in decisions if d.tier == DedupTier.AUTO_MERGE}
        dedup_stat = {"auto_merge": 0, "need_review": 0, "keep_both": 0}
        for d in decisions:
            dedup_stat[d.tier.value] += 1
        kept = [c for c in chunks if c.chunk_id not in merged_ids]

        records = [{
            "chunk_id": c.chunk_id,
            "text": c.text,
            "document": (c.meta or {}).get("document", "unknown"),
            "version": req.version,
        } for c in kept]
        written = writer.upsert(records) if writer is not None else []

        return schemas.IndexResponse(
            request_id=uuid.uuid4().hex[:12],
            routed=routed_cnt,
            rejected=rejected,
            chunks=len(chunks),
            written=list(written),
            dedup=dedup_stat,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ------------------------------------------------------------ 反馈回流
    def feedback(self, req: schemas.FeedbackRequest) -> schemas.FeedbackResponse:
        """维护班对答复的反馈落盘，供后续回流术语库 / 领域提示词。"""
        self._write_log("feedback.jsonl", {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "request_id": req.request_id,
            "helpful": req.helpful,
            "corrected_answer": req.corrected_answer,
            "comment": req.comment,
        })
        return schemas.FeedbackResponse(accepted=True, stored=True)
