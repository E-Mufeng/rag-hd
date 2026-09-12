"""
run_online_demo.py —— 在线问答管线演示（mock 模式）

演示：用户提问 -> C1 改写 -> C2 意图(双闸门+_DOMAIN_HINTS) -> C3 两路检索+RRF
      -> C4 精排(置信度聚合+阈值分级) -> C5 生成(Prompt+LLM 抽象)。
全程使用 mock 实现，不真检索、不真调大模型，仅打印各模块调用流程与决策。

运行：python examples/run_online_demo.py
"""
from __future__ import annotations

from pipeline_online import rewrite, intent, retrieve, rerank, generate
from mock_services import (
    mock_vectorstore, mock_bm25, mock_rerank, mock_llm, mock_intent,
)


def _to_candidates(chunks):
    return [(c.chunk_id, c.text) for c in chunks]


def ask(query: str):
    print("-" * 64)
    print(f"用户问题：{query}")

    # C1 改写（术语规范化；HyDE 默认关）
    rw = rewrite.rewrite(query, use_hyde=False)
    q = rw.rewritten
    print(f"[C1 改写] {rw.original} -> {q}  (hyde={rw.used_hyde})")

    # C2 意图（前置闸门 + _DOMAIN_HINTS）
    idec = intent.detect_intent(q, model=mock_intent.MockIntentModel(),
                                threshold=0.55)
    print(f"[C2 意图] label={idec.label} conf={idec.confidence} "
          f"pass_pre_gate={idec.passed_pre_gate} reason={idec.reason}")
    if not idec.passed_pre_gate:
        print(f"[C2 前置闸门] 未通过 -> 直接拒答：{idec.label}")
        return

    # C3 检索（dense + BM25 + RRF）
    vs = mock_vectorstore.MockVectorStore()
    bm = mock_bm25.MockBM25()
    fused = retrieve.retrieve(q, vs, bm, dense_top_k=5, bm25_top_k=5, rrf_k=60)
    print(f"[C3 检索] RRF 融合得到 {len(fused)} 个候选，"
          f"top1={fused[0].chunk_id}(rrf={fused[0].rrf_score})")

    # C4 精排（置信度聚合 + 阈值分级，检索后闸门）
    rr = mock_rerank.MockReranker()
    cdec = rerank.rerank(q, _to_candidates(fused), rr, top_n=5)
    print(f"[C4 精排] 置信度={cdec.confidence} 分级={cdec.level} "
          f"top1={cdec.top_results[0].chunk_id}({cdec.top_results[0].score})")

    # C5 生成（Prompt 模板 + LLM 抽象）
    ctxs = [r.text for r in cdec.top_results]
    llm = mock_llm.MockLLM()
    out = generate.generate(generate.GenerationInput(q, cdec, ctxs), llm)
    print(f"[C5 生成] {out[:80]}...")


def main():
    print("=" * 64)
    print("在线问答管线演示（mock）：双闸门 + 两路检索 + RRF + 置信度分级")
    print("=" * 64)
    for q in [
        "锅炉主蒸汽温度正常运行限额是多少",       # 业务查询 -> 高置信
        "#1机组 给水泵跳闸怎么处理",              # 业务查询（术语归一）-> 高/中
        "你好你是谁",                              # 闲聊 -> 拒答
        "今天股票行情怎么样",                      # 领域外 -> _DOMAIN_HINTS 前置拒答
    ]:
        ask(q)


if __name__ == "__main__":
    main()
