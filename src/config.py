"""全局运行时配置。

统一从环境变量 / `.env` 读取，全项目通过 `get_settings()` 获取单例配置对象。
本模块位于 `src/` 根目录，是所有包（parsing / agents / knowledge / graph / api）的
公共依赖，避免出现循环依赖。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 项目根目录（src/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """LCR-Agent 运行时配置。

    环境变量查找顺序：先匹配字段的 `validation_alias`，再匹配 `LCR_` 前缀名，
    最后回落到字段默认值。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="LCR_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ 应用
    app_env: str = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["http://localhost:5173"]

    # ------------------------------------------------------------------- LLM
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "LCR_OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "LCR_OPENAI_BASE_URL"),
    )
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # ------------------------------------------------------------- Embedding
    embedding_backend: str = "sentence-transformers"  # sentence-transformers | hashing
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dim: int = 1024

    # ------------------------------------------------------------- 向量存储
    chroma_dir: Path = PROJECT_ROOT / "data/index/chroma"
    chroma_collection: str = "cuad_clauses"

    # --------------------------------------------------------------- 检索
    top_k: int = 5
    top_k_vector: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    min_score: float = 0.2

    # --------------------------------------------------------------- 语料
    cuad_dir: Path = PROJECT_ROOT / "data/cuad"
    cuad_file: str = "CUAD_v1.json"
    configs_dir: Path = PROJECT_ROOT / "configs"
    default_config: str = "default.yaml"

    # --------------------------------------------------------------- 切分
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_encoding: str = "cl100k_base"

    # -------------------------------------------------------------- 派生属性
    @property
    def is_dev(self) -> bool:
        return self.app_env.lower() in {"dev", "development", "local"}

    @property
    def cuad_path(self) -> Path:
        return self.cuad_dir / self.cuad_file

    @property
    def has_llm(self) -> bool:
        """是否配置了可用的 LLM 凭据（否则走离线 stub 模型）。"""
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内共享的配置单例。"""
    return Settings()


def reload_settings() -> Settings:
    """清空缓存并重新加载配置（测试 / 热更新场景使用）。"""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "reload_settings"]
