"""Application configuration loaded from environment variables.

ILMU_API_KEY is the single environment variable for YTL ILMU access across the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TENDER_OS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = "postgresql://tenderos:tenderos@127.0.0.1:5439/tenderos"
    db_pool_min_size: int = Field(default=1, ge=1, le=50)
    db_pool_max_size: int = Field(default=10, ge=1, le=100)
    snapshot_path: str = "data/tenders_public.jsonl"
    embeddings_path: str = "data/embeddings.f32"

    ilmu_api_key: str | None = Field(default=None, validation_alias="ILMU_API_KEY")
    ilmu_base_url: str = "https://api.ilmu.ai/v1"
    ilmu_embed_model: str = "bge-m3"
    ilmu_embed_dim: int = 1024
    ilmu_rerank_model: str = "bge-reranker"
    ilmu_chat_primary_model: str = "ilmu-mini-v3.3"
    ilmu_chat_fallback_model: str = "ilmu-v3.1"
    ilmu_timeout_seconds: float = Field(default=15.0, gt=0.5, le=60.0)
    ilmu_rerank_timeout_seconds: float = Field(default=5.0, gt=0.5, le=30.0)

    rerank_floor: float = Field(default=0.04, ge=0.0, le=1.0)
    daily_cap_myr: float = Field(default=5.0, ge=0.0)
    rate_limit_rpm: int = Field(default=60, ge=1, le=10000)
    rrf_k: int = Field(default=60, ge=1)
    candidate_pool_size: int = Field(default=50, ge=5, le=200)
    rerank_pool_size: int = Field(default=20, ge=1, le=50)

    @property
    def daily_cap_micro_rm(self) -> int:
        return int(round(self.daily_cap_myr * 1_000_000))


@lru_cache(maxsize=4)
def get_settings() -> Settings:
    return Settings()
