"""
config.py —— 服务配置加载（base + 部署形态覆盖 + 环境变量）

加载顺序（后者覆盖前者）：
    1) configs/base.yaml                共享默认（阈值 / 检索参数 / 模型服务 URL）
    2) configs/{deployment.mode}.yaml   部署形态覆盖（docker.yaml：容器内服务名地址）
    3) 环境变量 RAG_*                   编排层注入（最高优先级，容器化部署的标准做法）

环境变量设计原则：**地址类参数一律走环境变量**，配置文件只留占位与默认值。
这样同一份镜像在试点机、扩容后的集群、测试环境之间漂移时不用改镜像内容，
只需改 compose 的 environment 段，符合「配置与镜像分离」的交付习惯。

    RAG_CONFIG_DIR        配置目录（默认 <repo>/configs），容器内可挂载覆盖
    RAG_USE_MOCK          覆盖 run.use_mock（true / false）
    RAG_MILVUS_URI        storage.vectorstore.uri
    RAG_OPENSEARCH_URI    storage.bm25.uri
    RAG_LLM_BASE_URL      models.llm.base_url
    RAG_LLM_MODEL         models.llm.model
    RAG_EMBEDDER_URL      models.embedder.service_url
    RAG_INTENT_URL        models.intent.service_url
    RAG_RERANKER_URL      models.reranker.service_url
    RAG_LAYOUT_URL        models.layout.service_url
    RAG_LOG_DIR           运行时日志目录（拒答 / 低置信 / 反馈落盘）
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import yaml

# 环境变量 -> 配置内路径（点号路径），None 表示需自行解析
_ENV_MAP: List[Tuple[str, str]] = [
    ("RAG_MILVUS_URI", "storage.vectorstore.uri"),
    ("RAG_OPENSEARCH_URI", "storage.bm25.uri"),
    ("RAG_LLM_BASE_URL", "models.llm.base_url"),
    ("RAG_LLM_MODEL", "models.llm.model"),
    ("RAG_EMBEDDER_URL", "models.embedder.service_url"),
    ("RAG_INTENT_URL", "models.intent.service_url"),
    ("RAG_RERANKER_URL", "models.reranker.service_url"),
    ("RAG_LAYOUT_URL", "models.layout.service_url"),
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base，dict 逐层下钻，其余类型直接替换。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """按点号路径写入（不存在则逐层创建）。"""
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_dir: str = None) -> Dict[str, Any]:
    """
    加载并合并配置，返回可直接喂给 factory.build_backends() 的 dict。

    真实部署注意：`deployment.mode` 决定叠加哪个部署覆盖文件；本框架固定为 docker。
    """
    cfg_dir = config_dir or os.environ.get("RAG_CONFIG_DIR") or os.path.join(_repo_root(), "configs")

    with open(os.path.join(cfg_dir, "base.yaml"), "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    mode = cfg.get("deployment", {}).get("mode", "docker")
    overlay = os.path.join(cfg_dir, f"{mode}.yaml")
    if os.path.exists(overlay):
        with open(overlay, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    # 环境变量覆盖（最高优先级）
    if os.environ.get("RAG_USE_MOCK") is not None:
        raw = os.environ["RAG_USE_MOCK"].strip().lower()
        cfg.setdefault("run", {})["use_mock"] = raw in ("1", "true", "yes", "on")
    for env_key, path in _ENV_MAP:
        val = os.environ.get(env_key)
        if val:
            _set_by_path(cfg, path, val)

    # 运行时日志目录（拒答 / 低置信 / 反馈）
    log_dir = os.environ.get("RAG_LOG_DIR")
    if log_dir:
        cfg.setdefault("logging", {})["dir"] = log_dir

    return cfg
