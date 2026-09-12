"""
run_with_factory.py —— 用 factory 装配后端、跑通在线问答链路（演示 mock/real 切换）

演示重点：业务代码从不 new 具体中间件，只通过 factory.build_backends(cfg) 拿抽象实例。
切换 mock/real 只改 configs 里的 run.use_mock（或本文件入参），业务零改动。

运行（mock 模式，无需任何服务）：
    PYTHONPATH=. python examples/run_with_factory.py
"""
from __future__ import annotations

import os
import yaml

from pipeline_online import factory
from pipeline_online.rewrite import rewrite
from pipeline_online.intent import detect_intent
from pipeline_online.retrieve import retrieve
from pipeline_online.rerank import rerank
from pipeline_online.generate import generate, GenerationInput, build_prompt


def _load_cfg() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "configs", "base.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(use_mock: bool = True):
    cfg = _load_cfg()
    backends = factory.build_backends(cfg, use_mock=use_mock)
    print(f"[factory] 装配后端：mode={cfg['deployment']['mode']}  "
          f"use_mock={use_mock}")
    print(f"          vectorstore={type(backends['vectorstore']).__name__}  "
          f"bm25={type(backends['bm25']).__name__}  "
          f"llm={type(backends['llm']).__name__}  "
          f"reranker={type(backends['reranker']).__name__}  "
          f"intent={type(backends['intent']).__name__}")

    queries = [
        "锅炉主蒸汽温度超温怎么处理？",
        "汽轮机润滑油压力低报警值是多少？",
        "今天天气怎么样？",            # 领域外，应前置拒答
    ]
    for q in queries:
        print("\n" + "=" * 60)
        print(f"用户问题：{q}")
        rw = rewrite(q)
        print(f"  改写后：{rw.rewritten}")
        intent = detect_intent(rw.rewritten, model=backends["intent"],
                               threshold=cfg["intent"]["threshold"])
        if not intent.passed_pre_gate:
            print(f"  [意图前置闸门] 拦截：{intent.label}（conf={intent.confidence}）")
            continue
        cands = retrieve(rw.rewritten, backends["vectorstore"], backends["bm25"],
                         dense_top_k=cfg["retrieval"]["dense_top_k"],
                         bm25_top_k=cfg["retrieval"]["bm25_top_k"],
                         rrf_k=cfg["retrieval"]["rrf_k"])
        cands = [(c.chunk_id, c.text) for c in cands[:cfg["retrieval"]["final_top_k"]]]
        decision = rerank(rw.rewritten, cands, backends["reranker"],
                          top_n=cfg["retrieval"]["final_top_k"])
        print(f"  [置信度闸门] level={decision.level}  conf={decision.confidence}")
        if decision.level == "low":
            print("  [生成] 强制拒答（低置信）")
            continue
        ans = generate(GenerationInput(q, decision, [t for _, t in cands]),
                       client=backends["llm"])
        print(f"  [生成] {ans[:80]}...")


if __name__ == "__main__":
    # use_mock=False 会装配真实后端（需内网服务就绪，否则调用时抛连接错误，属预期）
    main(use_mock=True)
