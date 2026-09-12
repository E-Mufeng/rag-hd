"""
intent.py —— C2 意图识别（MiniLM 抽象 + 规则兜底 + 双闸门 + _DOMAIN_HINTS + 拒答日志）

职责：判断用户问题是否属于火电运维领域、属于哪类意图（procedure/fault/chitchat/out_of_domain）。
机制：
    1) _DOMAIN_HINTS 全局前置守卫：任何问题先过「领域提示词」，若完全无领域信号
       且命中明显领域外关键词，可直接前置拒答（尽早拦截闲聊/无关问题，省检索资源）。
    2) MiniLM 轻模型为主：调用内网句向量服务做语义意图分类。
    3) 规则兜底：MiniLM 置信度不足 / 服务不可用时，退回关键词规则判定。
    4) 双闸门（与下游置信度闸门相互独立）：
        - 意图前置闸门：意图置信度 < threshold -> 不进入检索，直接拒答或转人工；
        - 检索后置信度闸门：在 rerank.py 中二次把关（见 rerank.py）。
    5) 拒答 / 低置信请求日志落盘：供维护班回流优化 _DOMAIN_HINTS 与术语库。

所属链路：在线 C2。
对接抽象接口：IntentModel（业务上层只依赖它）。

【仅框架演示，真实部署替换】
    MiniLMIntent 真实实现调用内网 embedding 服务得到句向量并做分类头推断；
    本文件提供 MockIntentModel（在 mock_services/mock_intent.py）作离线演示。
    不写任何模型下载 / 本地权重加载代码。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from mock_services.mock_intent import MockIntentModel


# 意图模板（真实 MiniLM 分类用的 4 类语义锚点；mock 规则兜底用 _KEYWORDS）
# 说明：MiniLM 句向量 + 模板最近邻分类，比关键词更抗「同义改写」，是真实部署的主路径。
_INTENT_TEMPLATES = {
    "procedure": "火电运行操作规程、参数限额、启停操作、系统操作方法",
    "fault": "火电设备故障、异常、事故处理、泄漏跳闸振动超温停机",
    "chitchat": "你好谢谢打招呼闲聊寒暄",
    "out_of_domain": "与火电运维无关的问题如足球股票菜谱旅游",
}

# —— _DOMAIN_HINTS：领域提示守卫（全局前置）——
# 用途：在意图模型之前先扫一遍，命中明显领域外词直接前置拒答；
#       命中强领域信号则给 MiniLM / 规则一个正向偏置。
_DOMAIN_HINTS = {
    "strong_in": ["锅炉", "汽轮机", "发电机", "给水泵", "主蒸汽", "脱硫", "脱硝",
                  "运行规程", "事故处理", "限额", "跳闸", "超温", "振动", "泄漏",
                  "#1机组", "#2机组", "定子", "润滑油", "浆液", "吸收塔"],
    "out_of_domain": ["足球", "股票", "菜谱", "旅游", "娱乐", "天气", "电影",
                     "相亲", "游戏", "房价"],
}

GATE_THRESHOLD = 0.55   # 意图前置闸门阈值（可由 configs.intent.threshold 覆盖）


@dataclass
class IntentDecision:
    label: str                 # procedure | fault | chitchat | out_of_domain | rejected
    confidence: float
    passed_pre_gate: bool      # 是否通过意图前置闸门
    reason: str = ""


class IntentModel(ABC):
    """意图模型抽象接口。"""

    @abstractmethod
    def classify(self, text: str) -> "IntentDecision":
        ...


class MiniLMIntent(IntentModel):
    """
    MiniLM 意图分类（真实实现封装）。

    【真实部署】调用内网 MiniLM embedding 服务得到句向量，再用模板最近邻分类
        （见 backends_real.MiniLMIntentService）。业务上层在真实部署时通过
        factory.build_backends() 直接拿到 MiniLMIntentService 实例注入本模块；
        此处保留同名类以便兼容旧调用，内部委托给真实服务实现。
    不写任何模型下载 / 本地权重加载代码；仅持有 service_url，调用时才发请求。
    """
    def __init__(self, service_url: str = "http://127.0.0.1:8002/v1/embeddings"):
        from pipeline_online.backends_real import MiniLMIntentService
        self._impl = MiniLMIntentService(service_url, templates=_INTENT_TEMPLATES)

    def classify(self, text: str) -> IntentDecision:
        return self._impl.classify(text)


def _rule_fallback(text: str) -> IntentDecision:
    """规则兜底：MiniLM 不可用 / 低置信时按关键词判定。"""
    from mock_services.mock_intent import _KEYWORDS
    best, cnt = "out_of_domain", 0
    for label, kws in _KEYWORDS.items():
        c = sum(1 for k in kws if k in text)
        if c > cnt:
            best, cnt = label, c
    conf = min(0.9, 0.4 + 0.15 * cnt) if cnt else 0.10
    return IntentDecision(label=best, confidence=conf, passed_pre_gate=False,
                          reason="rule_fallback")


def _domain_pre_guard(text: str) -> Optional[str]:
    """_DOMAIN_HINTS 前置守卫：返回 'reject' / 'bias_in' / None。"""
    for w in _DOMAIN_HINTS["out_of_domain"]:
        if w in text:
            return "reject"
    for w in _DOMAIN_HINTS["strong_in"]:
        if w in text:
            return "bias_in"
    return None


def log_reject(text: str, decision: IntentDecision, log_dir: str = "./data/logs") -> None:
    """拒答 / 低置信请求落盘（供维护班回流优化）。"""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "reject.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "text": text, "label": decision.label,
            "conf": decision.confidence, "reason": decision.reason,
        }, ensure_ascii=False) + "\n")


def detect_intent(text: str, model: Optional[IntentModel] = None,
                  threshold: float = GATE_THRESHOLD,
                  log_dir: str = "./data/logs") -> IntentDecision:
    """
    意图识别主入口（双闸门之「前置闸门」）。

    流程：
      1) _DOMAIN_HINTS 前置守卫 -> 明显领域外直接 reject（省检索）；
      2) MiniLM 模型分类（无模型则规则兜底）；
      3) 置信度 < threshold -> 前置拒答（passed_pre_gate=False）；
         否则放行进入检索阶段。
    """
    guard = _domain_pre_guard(text)
    if guard == "reject":
        d = IntentDecision(label="out_of_domain", confidence=0.95,
                           passed_pre_gate=False, reason="_DOMAIN_HINTS_pre_reject")
        log_reject(text, d, log_dir)
        return d

    if model is None:
        # 无模型注入时默认用 mock（离线演示）；真实部署传入 MiniLMIntent
        model = MockIntentModel()
    try:
        d = model.classify(text)
    except NotImplementedError:
        d = _rule_fallback(text)

    passed = d.confidence >= threshold
    d.passed_pre_gate = passed
    d.reason = (d.reason or "minilm") + ("|pass" if passed else "|below_threshold")
    if not passed:
        log_reject(text, d, log_dir)
    return d
