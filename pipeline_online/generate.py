"""
generate.py —— C5 生成（Prompt 模板 + LLM 抽象客户端）

职责：根据置信度决策与精排结果构造 Prompt，调用 LLM 生成最终回答。
Prompt 模板要点（幻觉抑制 / 溯源 / 强制拒答）：
    - 角色设定：火电运维知识库助手，仅依据提供资料作答；
    - 资料充足性校验：资料不足/置信度低时，必须如实说明并拒答或标注；
    - 幻觉抑制：不得编造规程参数，未知即说未知；
    - 来源可回溯：回答需标注引用来源（章节/页码）；
    - 强制拒答：置信度 < 0.4 时不生成，直接返回拒答话术。
所属链路：在线 C5。
对接抽象接口：LLMClient（vLLM OpenAI 兼容实现 + MockLLM）。

【仅框架演示，真实部署替换】
    真实实现 VLLMClient 调用内网 vLLM 服务：
        base_url: http://vllm:8000/v1  model: Qwen3-14B
    不写任何模型下载代码；本框架用 MockLLM 演示流程。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from pipeline_online.rerank import ConfidenceDecision


@dataclass
class GenerationInput:
    query: str
    decision: ConfidenceDecision
    contexts: List[str]


class LLMClient(ABC):
    """LLM 抽象客户端（vLLM OpenAI 兼容）。"""
    @abstractmethod
    def generate(self, prompt: str, contexts: List[str], temperature: float = 0.1):
        ...


def build_prompt(query: str, contexts: List[str], decision: ConfidenceDecision) -> str:
    """构造系统 Prompt（含角色设定 / 幻觉抑制 / 溯源 / 强制拒答）。"""
    if decision.level == "low":
        return ("【系统】当前检索置信度不足，依据强制拒答策略，不得编造回答，"
                "直接告知用户无法基于现有资料作答，并建议联系运维班组核实。")
    level_hint = ("（高置信：直接作答并标注来源）" if decision.level == "high"
                  else "（中置信：作答并明确提示『资料置信度有限，请核实』）")
    ctx_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "（无可用资料）"
    return (
        "你是一名火电运维知识库问答助手，仅依据下方【资料】作答，不得编造规程参数。\n"
        "要求：① 回答需标注引用来源（章节/页码）；② 资料不足时如实说明未知；\n"
        "③ 严禁幻觉。\n"
        f"用户问题：{query}\n"
        f"【资料】\n{ctx_block}\n"
        f"{level_hint}\n请基于资料作答："
    )


def generate(input: GenerationInput, client: LLMClient, temperature: float = 0.1) -> str:
    """
    生成入口：构造 Prompt -> 调用 LLM。
    - low 置信：走强制拒答话术（不调模型）；
    - 其他：调用 LLM 生成，并透传来源片段用于溯源。
    """
    prompt = build_prompt(input.query, input.contexts, input.decision)
    if input.decision.level == "low":
        return prompt  # 强制拒答，不调用模型
    return client.generate(prompt, input.contexts, temperature=temperature)
