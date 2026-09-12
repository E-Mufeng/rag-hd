"""
mock_llm —— 生成大模型（vLLM / Qwen）模拟实现

职责：模拟 LLM 问答生成接口（Prompt -> 回答）。
所属链路：在线 C5 生成。
对接抽象接口：pipeline_online/generate.py 中的 LLMClient 抽象基类。

【仅框架演示，真实部署替换】
    真实实现（generate.py 内的 VLLMClient）调用内网 vLLM OpenAI 兼容服务：
        base_url: http://vllm:8000/v1  （docker）
        model:    Qwen3-14B
    本 Mock 不调用任何模型，仅按模板回显来源片段、输出演示性回答。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResponse:
    answer: str
    sources: List[str]
    finish_reason: str


class MockLLM:
    """
    模拟 LLM 生成。真实部署替换为 VLLMClient（见 generate.py）。
    generate(prompt, contexts) -> str（最终回答文本）
    """

    def __init__(self, model: str = "mock-llm"):
        self.model = model

    def generate(self, prompt: str, contexts: List[str],
                 temperature: float = 0.1) -> str:
        if not contexts:
            return "（拒答）未检索到足够可信的资料，建议联系运维班组核实。"
        # 演示性回答：拼接来源片段首句，模拟「基于资料作答 + 溯源」
        head = contexts[0][:60]
        return (
            f"【模拟回答】根据检索到的运行规程/事故处理资料：{head}……"
            f"（本回答由 MockLLM 生成，仅用于演示流程，未调用真实大模型）"
        )
