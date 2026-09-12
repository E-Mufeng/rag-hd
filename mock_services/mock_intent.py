"""
mock_intent —— 意图识别（MiniLM）模拟实现

职责：模拟 MiniLM 轻模型意图分类接口。
所属链路：在线 C2 意图识别（前置闸门）。
对接抽象接口：pipeline_online/intent.py 中的 IntentModel 抽象基类。

【仅框架演示，真实部署替换】
    真实实现（intent.py 内的 MiniLMIntent）调用内网 embedding 服务得到句向量，
    再用轻量分类头/规则判定意图（service_url: http://127.0.0.1:8002/v1/embeddings）。
    本 Mock 仅用关键词规则确定性返回意图标签与置信度，不加载任何权重。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class IntentResult:
    label: str          # procedure | fault | chitchat | out_of_domain
    confidence: float
    evidence: str = ""
    reason: str = ""    # 供 intent.py 双闸门记录判定来源


# 意图关键词（演示用，覆盖常见火电术语；真实部署由 MiniLMIntentService 句向量分类，不依赖此表）
_KEYWORDS = {
    "procedure": ["规程", "限额", "运行", "操作", "参数", "启动", "停运", "汽轮机",
                  "润滑油", "压力", "报警", "温度", "流量", "设备", "系统", "负荷"],
    "fault": ["故障", "异常", "事故", "泄漏", "跳闸", "振动", "超温", "停机", "损坏", "断油"],
    "chitchat": ["你好", "谢谢", "你是谁", "天气", "打招呼"],
    "out_of_domain": ["足球", "股票", "菜谱", "旅游", "娱乐", "电影"],
}


class MockIntentModel:
    """
    模拟 MiniLM 意图分类。

    classify(text) -> IntentResult，按关键词命中最多者判意图，
    置信度由命中词比例近似。真实部署替换为 MiniLMIntent（句向量 + 分类头）。
    """

    def classify(self, text: str) -> IntentResult:
        best_label, best_cnt, best_ev = "out_of_domain", 0, ""
        for label, kws in _KEYWORDS.items():
            cnt = sum(1 for k in kws if k in text)
            if cnt > best_cnt:
                best_label, best_cnt, best_ev = label, cnt, kws[cnt - 1]
        # 命中越多置信度越高（演示用），无命中给极低分走拒答
        conf = round(min(0.95, 0.4 + 0.15 * best_cnt), 2) if best_cnt else 0.10
        return IntentResult(label=best_label, confidence=conf, evidence=best_ev)

    def embed(self, text: str) -> List[float]:
        """模拟句向量（真实部署调用内网 MiniLM embedding 服务）。"""
        import hashlib
        return [int.from_bytes(hashlib.md5(f"{text}#{i}".encode()).digest()[:4], "big") / 0xFFFFFFFF
                for i in range(16)]
